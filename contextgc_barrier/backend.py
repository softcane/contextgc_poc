from dataclasses import dataclass
from typing import Any, Dict, List, Protocol


class ContextBackend(Protocol):
    def count_tokens(self, messages: List[Dict[str, str]], model: str) -> int:
        ...

    def create(self, model: str, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        ...


@dataclass
class BackendMessage:
    content: str


@dataclass
class BackendChoice:
    message: BackendMessage


@dataclass
class BackendResponse:
    choices: List[BackendChoice]


def make_response(content: str) -> BackendResponse:
    return BackendResponse(choices=[BackendChoice(message=BackendMessage(content=content))])
