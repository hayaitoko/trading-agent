from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

Role = Literal["user", "assistant", "tool", "system"]

# Rough char-to-token ratio. English averages ~4, but 3 is conservative
# and matches the user's spec.
CHARS_PER_TOKEN = 3


@dataclass
class ChatMessage:
    role: Role
    content: str = ""
    images: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None
    model: str | None = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat(timespec="seconds")

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "images": self.images,
            "tool_calls": self.tool_calls,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "model": self.model,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        return cls(
            role=data["role"],
            content=data.get("content", "") or "",
            images=data.get("images") or [],
            tool_calls=data.get("tool_calls") or [],
            tool_call_id=data.get("tool_call_id"),
            tool_name=data.get("tool_name"),
            model=data.get("model"),
            timestamp=data.get("timestamp", ""),
        )


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display: str
    context_limit: int
    is_anthropic: bool = False


# Hardcoded defaults. Limits are approximate; edit as needed.
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("anthropic/claude-opus-4.7", "Claude Opus 4.7", 200_000, is_anthropic=True),
    ModelSpec("anthropic/claude-sonnet-4.6", "Claude Sonnet 4.6", 200_000, is_anthropic=True),
    ModelSpec("anthropic/claude-haiku-4.5", "Claude Haiku 4.5", 200_000, is_anthropic=True),
    ModelSpec("openai/gpt-5", "GPT-5", 400_000),
    ModelSpec("x-ai/grok-4", "Grok 4", 256_000),
    ModelSpec("google/gemini-3.1-pro", "Gemini 3.1 Pro", 2_000_000),
    ModelSpec("deepseek/deepseek-v4-pro", "DeepSeek v4 Pro", 128_000),
    ModelSpec("moonshotai/kimi-k2.6", "Kimi K2.6", 200_000),
)

DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"


def find_model(model_id: str) -> ModelSpec | None:
    for m in MODELS:
        if m.id == model_id:
            return m
    return None


def estimate_tokens(messages: list[ChatMessage]) -> int:
    chars = 0
    for m in messages:
        chars += len(m.content or "")
        for tc in m.tool_calls:
            chars += len(str(tc))
        # Rough image-token estimate. Real per-tile math is provider-specific;
        # this is a placeholder so the UI counter isn't blind to images.
        chars += len(m.images) * 1000 * CHARS_PER_TOKEN
    return chars // CHARS_PER_TOKEN
