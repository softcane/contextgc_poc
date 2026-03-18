from typing import Any, Dict, List, Optional
from .backend import make_response


class MLXBackend:
    def __init__(self, default_model: Optional[str] = None) -> None:
        self.default_model = default_model
        self._loaded_model_name: Optional[str] = None
        self._model = None
        self._tokenizer = None
        self._token_count_cache: dict[tuple[str, tuple[tuple[str, str], ...]], int] = {}

    def count_tokens(self, messages: List[Dict[str, str]], model: str) -> int:
        target = model or self.default_model
        if target is None:
            raise ValueError("MLXBackend requires a model name")

        cache_key = (
            target,
            tuple(
                (
                    str(message.get("role", "")),
                    str(message.get("content", "")),
                )
                for message in messages
            ),
        )
        if cache_key in self._token_count_cache:
            return self._token_count_cache[cache_key]

        _, tokenizer = self._ensure_loaded(model)
        if getattr(tokenizer, "chat_template", None) is not None:
            try:
                token_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    enable_thinking=False,
                )
                count = len(token_ids)
                self._remember_token_count(cache_key, count)
                return count
            except Exception:
                pass

        prompt = self._fallback_prompt(messages)
        count = len(tokenizer.encode(prompt))
        self._remember_token_count(cache_key, count)
        return count

    def create(self, model: str, messages: List[Dict[str, str]], **kwargs: Any):
        mlx_model, tokenizer = self._ensure_loaded(model)
        prompt = self._format_prompt(tokenizer, messages)

        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        temperature = kwargs.get("temperature", 0.0)

        text = generate(
            mlx_model,
            tokenizer,
            prompt=prompt,
            max_tokens=kwargs.get("max_tokens", 800),
            sampler=make_sampler(temp=temperature),
            verbose=False,
        )
        return make_response(text.strip())

    def model_max_length(self, model: str) -> int:
        _, tokenizer = self._ensure_loaded(model)
        return int(getattr(tokenizer, "model_max_length", 0) or 0)

    def _ensure_loaded(self, model: str):
        target = model or self.default_model
        if target is None:
            raise ValueError("MLXBackend requires a model name")

        if self._loaded_model_name != target:
            from mlx_lm import load

            self._model, self._tokenizer = load(target)
            self._loaded_model_name = target
            self._token_count_cache.clear()

        return self._model, self._tokenizer

    def _format_prompt(self, tokenizer, messages: List[Dict[str, str]]) -> str:
        if getattr(tokenizer, "chat_template", None) is not None:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )

        return self._fallback_prompt(messages)

    def _fallback_prompt(self, messages: List[Dict[str, str]]) -> str:
        lines = []
        for message in messages:
            lines.append(f"{message['role'].upper()}: {message['content']}")
        lines.append("ASSISTANT:")
        return "\n".join(lines)

    def _remember_token_count(
        self,
        cache_key: tuple[str, tuple[tuple[str, str], ...]],
        count: int,
    ) -> None:
        if len(self._token_count_cache) >= 2048:
            self._token_count_cache.clear()
        self._token_count_cache[cache_key] = count
