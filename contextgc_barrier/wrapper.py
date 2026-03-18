from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from .barrier import WriteBarrier
from .chunker import chunk_message
from .extractor import extract
from .registry import (
    ChunkRegistry,
    ContextMessage,
    ProtectionLevel,
)
from .scorer import score_all_chunks


@dataclass
class MessageSelection:
    selected_indexes: List[int]
    dropped_indexes: List[int]
    prompt_tokens: int


class ContextGCBarrier:
    def __init__(
        self,
        backend,
        window_budget: int = 128_000,
        response_budget: int = 800,
        strategy: str = "barrier",
        sticky_recent_messages: int = 4,
        citable_roles: tuple[str, ...] = ("user", "tool"),
    ) -> None:
        if strategy not in {"barrier", "recency"}:
            raise ValueError("strategy must be 'barrier' or 'recency'")

        self.backend = backend
        self.window_budget = window_budget
        self.response_budget = response_budget
        self.strategy = strategy
        self.sticky_recent_messages = sticky_recent_messages
        self.registry = ChunkRegistry()
        self.barrier = WriteBarrier(
            registry=self.registry,
            citable_roles=citable_roles,
        )
        self._turn = 0
        self._task_keywords: set[str] = set()
        self._registered_message_count = 0
        self._last_messages: List[Dict[str, str]] = []
        self._last_model: Optional[str] = None
        self._last_selection = MessageSelection([], [], 0)

    def chat(self, model: str, messages: List[Dict], **kwargs) -> Any:
        self._turn += 1
        self._last_model = model
        self._register_new_messages(model=model, messages=messages)
        self._update_task_keywords(messages)
        score_all_chunks(
            chunks=self.registry.all(),
            current_turn=self._turn,
            task_keywords=self._task_keywords,
        )

        selection = self._select_messages(model=model, messages=messages)
        selected_messages = [messages[index] for index in selection.selected_indexes]
        max_tokens = kwargs.pop("max_tokens", self.response_budget)
        response = self.backend.create(
            model=model,
            messages=selected_messages,
            max_tokens=max_tokens,
            temperature=kwargs.pop("temperature", 0.0),
            **kwargs,
        )

        response_text = response.choices[0].message.content or ""
        barrier_result = self.barrier.process(
            response_text=response_text,
            turn=self._turn,
            task_keywords=self._task_keywords,
        )
        score_all_chunks(
            chunks=self.registry.all(),
            current_turn=self._turn,
            task_keywords=self._task_keywords,
        )

        self._last_messages = list(messages)
        self._last_selection = selection
        response._cgc_barrier_result = barrier_result
        response._cgc_context_state = self.context_state()
        response._cgc_selected_messages = selected_messages
        response._cgc_prompt_tokens = selection.prompt_tokens
        response._cgc_strategy = self.strategy
        return response

    def context_state(self) -> Dict:
        messages = self._last_messages
        selected_indexes = set(self._last_selection.selected_indexes)
        protected_messages = self.registry.protected_message_ids()
        cited_messages = self.registry.cited_message_ids()
        stats = self.registry.stats()

        return {
            "turn": self._turn,
            "strategy": self.strategy,
            "window_budget": self.window_budget,
            "response_budget": self.response_budget,
            "prompt_tokens": self._last_selection.prompt_tokens,
            "total_messages": stats["total_messages"],
            "total_chunks": stats["total_chunks"],
            "total_tokens": stats["total_tokens"],
            "protected_message_count": len(protected_messages),
            "cited_message_count": len(cited_messages),
            "selected_messages": [
                self._message_snapshot(index, messages[index])
                for index in self._last_selection.selected_indexes
            ],
            "dropped_messages": [
                self._message_snapshot(index, message)
                for index, message in enumerate(messages)
                if index not in selected_indexes
            ],
            "barrier_summary": self.barrier.summary(),
        }

    def pin(self, message_index: int) -> None:
        message = self.registry.get_message_by_index(message_index)
        if message is None:
            return

        for chunk in self.registry.message_chunks(message.id):
            chunk.protection = ProtectionLevel.PINNED

    def report(self) -> str:
        state = self.context_state()
        lines = [
            f"=== ContextGC Barrier Report — Turn {state['turn']} ===",
            f"Strategy: {state['strategy']}",
            f"Prompt tokens: {state['prompt_tokens']} / {self.window_budget - self.response_budget}",
            f"Messages: {state['total_messages']} total, {len(state['selected_messages'])} selected, {len(state['dropped_messages'])} dropped",
            f"Protected messages: {state['protected_message_count']}, cited messages: {state['cited_message_count']}",
            "",
        ]

        if state["selected_messages"]:
            lines.append("Selected prompt:")
            for message in state["selected_messages"]:
                status = " protected" if message["protected"] else ""
                lines.append(
                    f"  [{message['index']}:{message['role']}, tokens={message['tokens']}, score={message['score']:.2f}, citations={message['citations']}{status}] "
                    f"{message['preview']}"
                )
            lines.append("")

        if state["dropped_messages"]:
            lines.append("Dropped messages:")
            for message in state["dropped_messages"][:10]:
                lines.append(
                    f"  [{message['index']}:{message['role']}, tokens={message['tokens']}, score={message['score']:.2f}, citations={message['citations']}] "
                    f"{message['preview']}"
                )

        return "\n".join(lines)

    def _register_new_messages(self, model: str, messages: List[Dict]) -> None:
        new_messages = messages[self._registered_message_count :]
        for index, message in enumerate(new_messages, start=self._registered_message_count):
            role = message.get("role", "user")
            content = message.get("content", "")
            if not isinstance(content, str) or not content:
                continue

            message_id = f"m_{index}"
            token_count = self.backend.count_tokens(
                messages=[{"role": role, "content": content}],
                model=model,
            )
            context_message = ContextMessage(
                id=message_id,
                index=index,
                role=role,
                content=content,
                turn=self._turn,
                tokens=token_count,
            )
            chunks = chunk_message(
                message_id=message_id,
                message_index=index,
                role=role,
                content=content,
                turn=self._turn,
                message_tokens=token_count,
            )
            self.registry.register_message(context_message, chunks)

        self._registered_message_count = len(messages)

    def _update_task_keywords(self, messages: List[Dict]) -> None:
        for message in reversed(messages):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                content = message["content"]
                if self._looks_like_bulk_payload(content):
                    continue
                self._task_keywords = extract(content).all_keywords
                if self._task_keywords:
                    return
        for message in reversed(messages):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                self._task_keywords = extract(message["content"]).all_keywords
                return
        self._task_keywords = set()

    def _select_messages(self, model: str, messages: List[Dict]) -> MessageSelection:
        if not messages:
            return MessageSelection([], [], 0)

        full_prompt_tokens = self.backend.count_tokens(
            messages=messages,
            model=model,
        )
        if full_prompt_tokens + self.response_budget <= self.window_budget:
            return MessageSelection(
                selected_indexes=list(range(len(messages))),
                dropped_indexes=[],
                prompt_tokens=full_prompt_tokens,
            )

        hard_indexes = set(index for index, message in enumerate(messages) if message.get("role") == "system")
        current_user_index = self._latest_user_index(messages)
        if current_user_index is not None:
            hard_indexes.add(current_user_index)

        recent_non_system = [
            index
            for index, message in enumerate(messages)
            if message.get("role") != "system"
        ]
        if self.sticky_recent_messages > 0:
            recent_non_system = recent_non_system[-self.sticky_recent_messages :]
        else:
            recent_non_system = []
        soft_anchor_indexes = [index for index in recent_non_system if index not in hard_indexes]

        selected = set(hard_indexes) | set(soft_anchor_indexes)
        selected = self._fit_soft_anchors(
            model=model,
            messages=messages,
            selected=selected,
            soft_anchor_indexes=soft_anchor_indexes,
        )

        candidate_indexes = [
            index for index in range(len(messages))
            if index not in selected
        ]
        ordered_candidates = self._ordered_candidates(candidate_indexes)

        for index in ordered_candidates:
            if self.strategy == "barrier" and not self._should_include_barrier_candidate(index):
                continue
            trial_indexes = sorted(selected | {index})
            prompt_tokens = self.backend.count_tokens(
                messages=[messages[i] for i in trial_indexes],
                model=model,
            )
            if prompt_tokens + self.response_budget <= self.window_budget:
                selected.add(index)

        selected_indexes = sorted(selected)
        dropped_indexes = [index for index in range(len(messages)) if index not in selected]
        prompt_tokens = self.backend.count_tokens(
            messages=[messages[index] for index in selected_indexes],
            model=model,
        )
        return MessageSelection(selected_indexes, dropped_indexes, prompt_tokens)

    def _fit_soft_anchors(
        self,
        model: str,
        messages: List[Dict],
        selected: set[int],
        soft_anchor_indexes: List[int],
    ) -> set[int]:
        prompt_tokens = self.backend.count_tokens(
            messages=[messages[index] for index in sorted(selected)],
            model=model,
        )
        if prompt_tokens + self.response_budget <= self.window_budget:
            return selected

        for index in sorted(soft_anchor_indexes):
            selected.discard(index)
            prompt_tokens = self.backend.count_tokens(
                messages=[messages[i] for i in sorted(selected)],
                model=model,
            )
            if prompt_tokens + self.response_budget <= self.window_budget:
                break
        return selected

    def _ordered_candidates(self, candidate_indexes: List[int]) -> List[int]:
        if self.strategy == "recency":
            return sorted(candidate_indexes, reverse=True)

        def barrier_key(index: int):
            message = self.registry.get_message_by_index(index)
            if message is None:
                return (0, 0.0, 0, index)
            return (
                1 if self.registry.message_is_protected(message.id) else 0,
                1 if self._message_has_task_overlap(message.id) else 0,
                self.registry.message_score(message.id),
                message.turn,
                index,
            )

        return sorted(candidate_indexes, key=barrier_key, reverse=True)

    def _should_include_barrier_candidate(self, index: int) -> bool:
        message = self.registry.get_message_by_index(index)
        if message is None:
            return False
        if self.registry.message_is_protected(message.id):
            return True
        return self._message_has_task_overlap(message.id)

    def _latest_user_index(self, messages: List[Dict]) -> Optional[int]:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user":
                return index
        return None

    def _message_has_task_overlap(self, message_id: str) -> bool:
        if not self._task_keywords:
            return False
        return any(
            bool(chunk.keywords & self._task_keywords)
            for chunk in self.registry.message_chunks(message_id)
        )

    def _message_snapshot(self, index: int, message: Dict[str, Any]) -> Dict[str, Any]:
        registry_message = self.registry.get_message_by_index(index)
        message_id = registry_message.id if registry_message else None
        return {
            "index": index,
            "role": message.get("role", "user"),
            "tokens": registry_message.tokens if registry_message else 0,
            "score": self.registry.message_score(message_id) if message_id else 0.0,
            "citations": self.registry.message_citations(message_id) if message_id else 0,
            "protected": self.registry.message_is_protected(message_id) if message_id else False,
            "preview": _preview(message.get("content", "")),
        }

    def _looks_like_bulk_payload(self, content: str) -> bool:
        return len(content) > 1_200 or content.count("\n") > 30


def _preview(content: str, limit: int = 80) -> str:
    if not isinstance(content, str):
        return ""
    if len(content) <= limit:
        return content
    return content[:limit] + "..."
