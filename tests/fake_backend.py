from typing import Any, Dict, List
from contextgc_barrier.backend import make_response

CRITICAL_DETAILS = [
    ("process_batch", "process_batch()"),
    ("batch_size > 1000", "batch_size > 1000"),
    ("numpy cache", "numpy cache"),
    ("v2.3", "v2.3"),
    ("src/batch_processor.py", "src/batch_processor.py"),
    ("200mb", "200MB"),
    ("cache.clear()", "cache.clear()"),
]


class FakeBackend:
    def count_tokens(self, messages: List[Dict[str, str]], model: str) -> int:
        return sum(self._message_tokens(message) for message in messages)

    def create(self, model: str, messages: List[Dict[str, str]], **kwargs: Any):
        prompt_text = "\n".join(
            str(message.get("content", "")).lower()
            for message in messages
        )
        last_user = next(
            (
                str(message.get("content", "")).lower()
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        )

        if "before i close this ticket" in last_user:
            found = [response_text for needle, response_text in CRITICAL_DETAILS if needle in prompt_text]
            if not found:
                return make_response("I lost the earlier context. I only remember a cache cleanup idea.")
            return make_response("Summary: " + "; ".join(found) + ".")

        if "memory leak happens specifically" in last_user:
            return make_response(
                "Initial read: process_batch() plus the numpy cache from v2.3 looks suspicious."
            )

        if "thread-safety" in last_user or "cache.clear()" in last_user:
            return make_response(
                "Use cache.clear() behind a lock, but the bug still points to process_batch()."
            )

        if "where exactly would you patch" in last_user:
            return make_response(
                "Patch src/batch_processor.py around process_batch() and clear the numpy cache."
            )

        if "log dump" in last_user:
            return make_response(
                "The logs are noisy, but process_batch() still matches the RSS growth pattern."
            )

        return make_response("I am still checking process_batch() and the numpy cache.")

    def _message_tokens(self, message: Dict[str, str]) -> int:
        content = str(message.get("content", ""))
        return len(content.split()) + 4
