from typing import Any, Dict, List
import re

from benchmark.tasks import CODING_VALUES, DEBUGGING_VALUES, DOCUMENT_VALUES, SUPPORT_VALUES
from contextgc_barrier.backend import make_response


def _flatten_aliases() -> list[str]:
    aliases: list[str] = []
    for pool_group in (DEBUGGING_VALUES, DOCUMENT_VALUES, CODING_VALUES, SUPPORT_VALUES):
        for pool in pool_group.values():
            for value_aliases in pool.values():
                aliases.extend(value_aliases)
    return sorted(set(aliases), key=len, reverse=True)


KNOWN_ALIASES = _flatten_aliases()


class FakeBackend:
    def count_tokens(self, messages: List[Dict[str, str]], model: str) -> int:
        return sum(self._message_tokens(message) for message in messages)

    def create(self, model: str, messages: List[Dict[str, str]], **kwargs: Any):
        prompt_text_raw = "\n".join(str(message.get("content", "")) for message in messages)
        prompt_text = prompt_text_raw.lower()
        last_user = next(
            (
                str(message.get("content", ""))
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        )
        detected = self._detected_aliases(prompt_text_raw)

        if "compress conversation state" in prompt_text or "working-memory bullets" in prompt_text:
            bullets = detected[:4] or ["earlier context compressed"]
            return make_response("Rolling summary:\n" + "\n".join(f"- {item}" for item in bullets))

        if self._looks_like_final_prompt(last_user.lower()):
            if not detected:
                return make_response("The active case details were lost in the noise.")
            return make_response("Final note: " + "; ".join(detected[:8]) + ".")

        if detected:
            return make_response("Active details: " + "; ".join(detected[:5]) + ".")

        return make_response("I only have a high-level read of the active case.")

    def _message_tokens(self, message: Dict[str, str]) -> int:
        content = str(message.get("content", ""))
        return len(content.split()) + 4

    def _detected_aliases(self, prompt_text: str) -> list[str]:
        found: list[tuple[int, str]] = []
        lowered = prompt_text.lower()
        for alias in KNOWN_ALIASES:
            pattern = re.escape(alias).replace(r"\ ", r"\s+")
            match = re.search(pattern, lowered, re.IGNORECASE)
            if match:
                found.append((match.start(), alias))
        found.sort(key=lambda item: (item[0], -len(item[1])))
        ordered: list[str] = []
        seen: set[str] = set()
        for _, alias in found:
            normalized = re.sub(r"[^a-z0-9@/._:-]+", " ", alias.lower()).strip()
            if normalized in seen:
                continue
            ordered.append(alias)
            seen.add(normalized)
        return ordered

    def _looks_like_final_prompt(self, last_user: str) -> bool:
        return any(
            phrase in last_user
            for phrase in (
                "release manager",
                "postmortem close note",
                "engineering handoff paragraph",
                "reply to counsel",
                "final policy answer",
                "compliance response",
                "release handoff paragraph",
                "handoff note for the active rollout",
                "draft the rollout summary",
                "support lead will send",
                "closing note for the active customer case",
                "final customer response",
            )
        )
