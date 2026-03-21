from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .backend import make_response
from .barrier import WriteBarrier
from .chunker import chunk_message
from .extractor import ExtractorMode, extract
from .registry import (
    ChunkRegistry,
    ContextMessage,
    ProtectionLevel,
)
from .scorer import score_all_chunks
from .summary import SummaryState, compose_summary_prompt, ensure_summary_fit


VALID_STRATEGIES = {"summary80", "barrier", "summary80_barrier"}


@dataclass
class MessageSelection:
    selected_indexes: List[int]
    dropped_indexes: List[int]
    prompt_tokens: int
    prompt_messages: List[Dict[str, str]]
    summary_active: bool = False
    summarized_through_index: int = 0
    summary_tokens: int = 0
    protected_exception_indexes: List[int] = field(default_factory=list)


class ContextGCBarrier:
    def __init__(
        self,
        backend,
        window_budget: int = 128_000,
        response_budget: int = 800,
        strategy: str = "barrier",
        sticky_recent_messages: int = 4,
        citable_roles: tuple[str, ...] = ("user", "tool"),
        citation_enabled: bool = True,
        extractor_mode: ExtractorMode = "spacy",
    ) -> None:
        if strategy not in VALID_STRATEGIES:
            raise ValueError("strategy must be 'summary80', 'barrier', or 'summary80_barrier'")

        self.backend = backend
        self.window_budget = window_budget
        self.response_budget = response_budget
        self.strategy = strategy
        self.sticky_recent_messages = sticky_recent_messages
        self.citation_enabled = citation_enabled
        self.extractor_mode = extractor_mode
        self.registry = ChunkRegistry()
        self.barrier = WriteBarrier(
            registry=self.registry,
            citable_roles=citable_roles,
            citation_enabled=citation_enabled,
            extractor_mode=extractor_mode,
        )
        self._turn = 0
        self._task_keywords: set[str] = set()
        self._registered_message_count = 0
        self._last_messages: List[Dict[str, str]] = []
        self._last_model: Optional[str] = None
        self._last_selection = MessageSelection([], [], 0, [])
        self._summary_state: Optional[SummaryState] = None
        self._replay_mode = False

    def chat(self, model: str, messages: List[Dict], **kwargs) -> Any:
        selection = self._begin_turn(model=model, messages=messages)
        max_tokens = kwargs.pop("max_tokens", self.response_budget)
        response = self.backend.create(
            model=model,
            messages=selection.prompt_messages,
            max_tokens=max_tokens,
            temperature=kwargs.pop("temperature", 0.0),
            **kwargs,
        )
        response_text = response.choices[0].message.content or ""
        return self._finish_turn(
            messages=messages,
            selection=selection,
            response=response,
            response_text=response_text,
        )

    def replay_turn(
        self,
        model: str,
        messages_before_reply: List[Dict[str, str]],
        reply_text: str,
        **kwargs: Any,
    ) -> Any:
        self._replay_mode = True
        try:
            selection = self._begin_turn(model=model, messages=messages_before_reply)
        finally:
            self._replay_mode = False
        response = make_response(reply_text)
        if "max_tokens" in kwargs:
            response._cgc_max_tokens = kwargs["max_tokens"]
        return self._finish_turn(
            messages=messages_before_reply,
            selection=selection,
            response=response,
            response_text=reply_text,
        )

    def context_state(self) -> Dict:
        messages = self._last_messages
        selected_indexes = set(self._last_selection.selected_indexes)
        protected_messages = self.registry.protected_message_ids()
        cited_messages = self.registry.cited_message_ids()
        protected_indexes = sorted(
            self.registry.get_message(message_id).index
            for message_id in protected_messages
            if self.registry.get_message(message_id) is not None
        )
        cited_indexes = sorted(
            self.registry.get_message(message_id).index
            for message_id in cited_messages
            if self.registry.get_message(message_id) is not None
        )
        stats = self.registry.stats()

        return {
            "turn": self._turn,
            "strategy": self.strategy,
            "window_budget": self.window_budget,
            "response_budget": self.response_budget,
            "citation_enabled": self.citation_enabled,
            "extractor_mode": self.extractor_mode,
            "prompt_tokens": self._last_selection.prompt_tokens,
            "total_messages": stats["total_messages"],
            "total_chunks": stats["total_chunks"],
            "total_tokens": stats["total_tokens"],
            "protected_message_count": len(protected_messages),
            "cited_message_count": len(cited_messages),
            "protected_message_indexes": protected_indexes,
            "cited_message_indexes": cited_indexes,
            "selected_messages": [
                self._message_snapshot(index, messages[index])
                for index in self._last_selection.selected_indexes
                if index < len(messages)
            ],
            "dropped_messages": [
                self._message_snapshot(index, message)
                for index, message in enumerate(messages)
                if index not in selected_indexes
            ],
            "summary_active": self._last_selection.summary_active,
            "summarized_through_index": self._last_selection.summarized_through_index,
            "summary_tokens": self._last_selection.summary_tokens,
            "protected_exception_indexes": list(self._last_selection.protected_exception_indexes),
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
            f"=== ContextGC Barrier Report - Turn {state['turn']} ===",
            f"Strategy: {state['strategy']}",
            f"Prompt tokens: {state['prompt_tokens']} / {self.window_budget - self.response_budget}",
            f"Messages: {state['total_messages']} total, {len(state['selected_messages'])} raw selected, {len(state['dropped_messages'])} dropped",
            f"Protected messages: {state['protected_message_count']}, cited messages: {state['cited_message_count']}",
        ]
        if state["summary_active"]:
            lines.append(
                "Summary: "
                f"active through raw index {state['summarized_through_index']}, "
                f"summary tokens={state['summary_tokens']}, "
                f"protected exceptions={state['protected_exception_indexes']}"
            )
        lines.append("")

        if state["selected_messages"]:
            lines.append("Selected raw messages:")
            for message in state["selected_messages"]:
                status = " protected" if message["protected"] else ""
                lines.append(
                    f"  [{message['index']}:{message['role']}, tokens={message['tokens']}, score={message['score']:.2f}, citations={message['citations']}{status}] "
                    f"{message['preview']}"
                )
            lines.append("")

        if state["dropped_messages"]:
            lines.append("Dropped raw messages:")
            for message in state["dropped_messages"][:10]:
                lines.append(
                    f"  [{message['index']}:{message['role']}, tokens={message['tokens']}, score={message['score']:.2f}, citations={message['citations']}] "
                    f"{message['preview']}"
                )

        return "\n".join(lines)

    def _begin_turn(self, model: str, messages: List[Dict]) -> MessageSelection:
        self._turn += 1
        self._last_model = model
        self._register_new_messages(model=model, messages=messages)
        self._update_task_keywords(messages)
        score_all_chunks(
            chunks=self.registry.all(),
            current_turn=self._turn,
            task_keywords=self._task_keywords,
        )
        return self._select_messages(model=model, messages=messages)

    def _finish_turn(
        self,
        *,
        messages: List[Dict],
        selection: MessageSelection,
        response: Any,
        response_text: str,
    ) -> Any:
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
        response._cgc_selected_messages = list(selection.prompt_messages)
        response._cgc_prompt_tokens = selection.prompt_tokens
        response._cgc_strategy = self.strategy
        return response

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
                extractor_mode=self.extractor_mode,
            )
            self.registry.register_message(context_message, chunks)

        self._registered_message_count = len(messages)

    def _update_task_keywords(self, messages: List[Dict]) -> None:
        for message in reversed(messages):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                content = message["content"]
                if self._looks_like_bulk_payload(content):
                    continue
                self._task_keywords = extract(content, mode=self.extractor_mode).all_keywords
                if self._task_keywords:
                    return
        for message in reversed(messages):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                self._task_keywords = extract(message["content"], mode=self.extractor_mode).all_keywords
                return
        self._task_keywords = set()

    def _select_messages(self, model: str, messages: List[Dict]) -> MessageSelection:
        if not messages:
            return MessageSelection([], [], 0, [])
        if self.strategy == "barrier":
            self._summary_state = None
            return self._select_barrier_messages(model=model, messages=messages)
        return self._select_summary_messages(model=model, messages=messages)

    def _select_barrier_messages(self, model: str, messages: List[Dict]) -> MessageSelection:
        full_prompt_tokens = self.backend.count_tokens(
            messages=messages,
            model=model,
        )
        if full_prompt_tokens + self.response_budget <= self.window_budget:
            return MessageSelection(
                selected_indexes=list(range(len(messages))),
                dropped_indexes=[],
                prompt_tokens=full_prompt_tokens,
                prompt_messages=[_as_prompt_message(message) for message in messages],
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
        ordered_candidates = self._ordered_barrier_candidates(candidate_indexes)

        for index in ordered_candidates:
            if not self._should_include_barrier_candidate(index):
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
        prompt_messages = [_as_prompt_message(messages[index]) for index in selected_indexes]
        prompt_tokens = self.backend.count_tokens(messages=prompt_messages, model=model)
        return MessageSelection(
            selected_indexes=selected_indexes,
            dropped_indexes=dropped_indexes,
            prompt_tokens=prompt_tokens,
            prompt_messages=prompt_messages,
        )

    def _select_summary_messages(self, model: str, messages: List[Dict]) -> MessageSelection:
        full_prompt_tokens = self.backend.count_tokens(messages=messages, model=model)
        if full_prompt_tokens + self.response_budget <= self.window_budget:
            self._summary_state = SummaryState(
                summary_message=None,
                summarized_through_index=0,
                tail_messages=[
                    {"index": index, "role": message.get("role", "user"), "content": message.get("content", "")}
                    for index, message in enumerate(messages)
                    if message.get("role") != "system"
                ],
            )
            return MessageSelection(
                selected_indexes=list(range(len(messages))),
                dropped_indexes=[],
                prompt_tokens=full_prompt_tokens,
                prompt_messages=[_as_prompt_message(message) for message in messages],
            )

        system_prompt = next(
            (str(message.get("content", "")) for message in messages if message.get("role") == "system"),
            "",
        )
        history = [
            {"index": index, "role": message.get("role", "user"), "content": str(message.get("content", ""))}
            for index, message in enumerate(messages)
            if message.get("role") != "system"
        ]
        usable_prompt_budget = self.window_budget - self.response_budget
        summary_trigger = int(usable_prompt_budget * 0.80)
        summary_cap = max(32, int(usable_prompt_budget * 0.15))
        prior_state = self._summary_state
        if prior_state is not None and prior_state.summarized_through_index > len(messages):
            prior_state = None
        summary_state = ensure_summary_fit(
            backend=self.backend,
            model_name=model,
            system_prompt=system_prompt,
            history=history,
            window_budget=self.window_budget,
            response_budget=self.response_budget,
            summary_trigger=summary_trigger,
            summary_cap=summary_cap,
            prior_state=prior_state,
            allow_generation=not self._replay_mode,
        )
        self._summary_state = summary_state

        system_indexes = [index for index, message in enumerate(messages) if message.get("role") == "system"]
        tail_indexes = [message["index"] for message in summary_state.tail_messages]
        protected_exception_indexes: list[int] = []

        if self.strategy == "summary80_barrier":
            protected_exception_indexes = self._protected_exception_indexes(
                messages=messages,
                summarized_through_index=summary_state.summarized_through_index,
                excluded_indexes=set(system_indexes) | set(tail_indexes),
            )
            prompt_messages, tail_indexes, protected_exception_indexes = self._fit_hybrid_prompt(
                model=model,
                messages=messages,
                system_prompt=system_prompt,
                summary_state=summary_state,
                tail_indexes=tail_indexes,
                protected_exception_indexes=protected_exception_indexes,
            )
        else:
            prompt_messages = compose_summary_prompt(system_prompt, summary_state.summary_message, summary_state.tail_messages)

        selected_indexes = sorted(set(system_indexes) | set(tail_indexes) | set(protected_exception_indexes))
        dropped_indexes = [index for index in range(len(messages)) if index not in selected_indexes]
        summary_tokens = 0
        if summary_state.summary_message is not None:
            summary_tokens = self.backend.count_tokens(
                [summary_state.summary_message],
                model=model,
            )
        prompt_tokens = self.backend.count_tokens(prompt_messages, model=model)
        return MessageSelection(
            selected_indexes=selected_indexes,
            dropped_indexes=dropped_indexes,
            prompt_tokens=prompt_tokens,
            prompt_messages=prompt_messages,
            summary_active=summary_state.summary_message is not None,
            summarized_through_index=summary_state.summarized_through_index,
            summary_tokens=summary_tokens,
            protected_exception_indexes=protected_exception_indexes,
        )

    def _fit_hybrid_prompt(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        summary_state: SummaryState,
        tail_indexes: List[int],
        protected_exception_indexes: List[int],
    ) -> tuple[List[Dict[str, str]], List[int], List[int]]:
        latest_user_index = self._latest_user_index(messages)
        surviving_tail = list(tail_indexes)
        surviving_exceptions = list(protected_exception_indexes)
        protected_tail_indexes = {
            index for index in tail_indexes
            if self._message_is_protected_or_cited(index)
        }

        prompt_messages = self._compose_hybrid_prompt(
            system_prompt=system_prompt,
            summary_message=summary_state.summary_message,
            messages=messages,
            protected_exception_indexes=surviving_exceptions,
            tail_indexes=surviving_tail,
        )
        while self.backend.count_tokens(prompt_messages, model=model) + self.response_budget > self.window_budget:
            removable_tail_index = next(
                (
                    index for index in surviving_tail
                    if index != latest_user_index and index not in protected_tail_indexes
                ),
                None,
            )
            if removable_tail_index is None:
                break
            surviving_tail.remove(removable_tail_index)
            prompt_messages = self._compose_hybrid_prompt(
                system_prompt=system_prompt,
                summary_message=summary_state.summary_message,
                messages=messages,
                protected_exception_indexes=surviving_exceptions,
                tail_indexes=surviving_tail,
            )

        if self.backend.count_tokens(prompt_messages, model=model) + self.response_budget <= self.window_budget:
            return prompt_messages, surviving_tail, surviving_exceptions

        drop_order = sorted(
            surviving_exceptions,
            key=self._protected_exception_priority,
            reverse=True,
        )
        while self.backend.count_tokens(prompt_messages, model=model) + self.response_budget > self.window_budget and drop_order:
            surviving_exceptions.remove(drop_order.pop())
            prompt_messages = self._compose_hybrid_prompt(
                system_prompt=system_prompt,
                summary_message=summary_state.summary_message,
                messages=messages,
                protected_exception_indexes=surviving_exceptions,
                tail_indexes=surviving_tail,
            )

        while self.backend.count_tokens(prompt_messages, model=model) + self.response_budget > self.window_budget:
            removable_tail_index = next(
                (index for index in surviving_tail if index != latest_user_index),
                None,
            )
            if removable_tail_index is None:
                break
            surviving_tail.remove(removable_tail_index)
            prompt_messages = self._compose_hybrid_prompt(
                system_prompt=system_prompt,
                summary_message=summary_state.summary_message,
                messages=messages,
                protected_exception_indexes=surviving_exceptions,
                tail_indexes=surviving_tail,
            )
        return prompt_messages, surviving_tail, surviving_exceptions

    def _compose_hybrid_prompt(
        self,
        *,
        system_prompt: str,
        summary_message: Optional[dict[str, str]],
        messages: List[Dict[str, Any]],
        protected_exception_indexes: List[int],
        tail_indexes: List[int],
    ) -> List[Dict[str, str]]:
        prompt_messages = [{"role": "system", "content": system_prompt}]
        if summary_message is not None:
            prompt_messages.append(summary_message)
        prompt_messages.extend(_as_prompt_message(messages[index]) for index in protected_exception_indexes)
        prompt_messages.extend(_as_prompt_message(messages[index]) for index in tail_indexes)
        return prompt_messages

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

    def _ordered_barrier_candidates(self, candidate_indexes: List[int]) -> List[int]:
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

    def _protected_exception_indexes(
        self,
        *,
        messages: List[Dict[str, Any]],
        summarized_through_index: int,
        excluded_indexes: set[int],
    ) -> List[int]:
        candidates: list[int] = []
        for index, message in enumerate(messages):
            if index in excluded_indexes or index > summarized_through_index:
                continue
            if message.get("role") not in {"user", "tool"}:
                continue
            if self._message_is_protected_or_cited(index):
                candidates.append(index)
        return candidates

    def _message_is_protected_or_cited(self, index: int) -> bool:
        registry_message = self.registry.get_message_by_index(index)
        if registry_message is None:
            return False
        return self.registry.message_is_protected(registry_message.id) or self.registry.message_citations(registry_message.id) > 0

    def _protected_exception_priority(self, index: int) -> tuple[int, float, int]:
        registry_message = self.registry.get_message_by_index(index)
        if registry_message is None:
            return (0, 0.0, index)
        return (
            self.registry.message_citations(registry_message.id),
            self.registry.message_score(registry_message.id),
            index,
        )

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


def _as_prompt_message(message: Dict[str, Any]) -> Dict[str, str]:
    return {
        "role": str(message.get("role", "user")),
        "content": str(message.get("content", "")),
    }


def _preview(content: str, limit: int = 80) -> str:
    if not isinstance(content, str):
        return ""
    if len(content) <= limit:
        return content
    return content[:limit] + "..."
