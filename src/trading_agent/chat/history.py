import json
from pathlib import Path

from trading_agent.chat.models import ChatMessage


def load_history(path: Path) -> list[ChatMessage]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [ChatMessage.from_dict(d) for d in raw]


def save_history(messages: list[ChatMessage], path: Path) -> None:
    path.write_text(
        json.dumps([m.to_dict() for m in messages], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def clear_history(path: Path) -> None:
    if path.exists():
        path.unlink()
