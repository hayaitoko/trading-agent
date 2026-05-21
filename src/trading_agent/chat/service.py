import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from trading_agent.chat import tools
from trading_agent.chat.client import call_model
from trading_agent.chat.history import clear_history, load_history, save_history
from trading_agent.chat.models import DEFAULT_MODEL, ChatMessage, find_model

if TYPE_CHECKING:
    from trading_agent.web.state import AppState

MAX_TOOL_ITERATIONS = 6

ModelCaller = Callable[..., Awaitable[dict[str, Any]]]


class ChatService:
    def __init__(
        self,
        state: "AppState",
        history_path: Path,
        *,
        model_caller: ModelCaller | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.state = state
        self.history_path = history_path
        self._model_caller = model_caller or call_model
        self._transport = transport

    def load(self) -> list[ChatMessage]:
        return load_history(self.history_path)

    def reset(self) -> None:
        clear_history(self.history_path)

    async def send(
        self, user_text: str, images: list[str], model_id: str | None = None
    ) -> list[ChatMessage]:
        model = find_model(model_id or DEFAULT_MODEL)
        if model is None:
            raise ValueError(f"unknown model: {model_id}")

        history = self.load()
        new_messages: list[ChatMessage] = [
            ChatMessage(role="user", content=user_text, images=images)
        ]

        for _ in range(MAX_TOOL_ITERATIONS):
            response = await self._model_caller(
                api_key=self.state.secrets.get("openrouter_api_key", ""),
                model=model,
                system_prompt=tools.SYSTEM_PROMPT,
                history=[*history, *new_messages],
                tools=tools.TOOL_SCHEMAS,
                transport=self._transport,
            )

            tool_calls = response.get("tool_calls") or []
            content = response.get("content") or ""

            if tool_calls:
                new_messages.append(ChatMessage(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                    model=model.id,
                ))
                for tc in tool_calls:
                    tool_result = await self._execute(tc)
                    new_messages.append(ChatMessage(
                        role="tool",
                        content=tool_result,
                        tool_call_id=tc.get("id"),
                        tool_name=(tc.get("function") or {}).get("name"),
                    ))
                continue

            new_messages.append(ChatMessage(
                role="assistant",
                content=content,
                model=model.id,
            ))
            break
        else:
            new_messages.append(ChatMessage(
                role="assistant",
                content="(stopped after hitting tool-iteration limit)",
                model=model.id,
            ))

        history.extend(new_messages)
        save_history(history, self.history_path)
        return history

    async def _execute(self, tool_call: dict) -> str:
        fn = tool_call.get("function") or {}
        name = fn.get("name") or ""
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            return json.dumps({"error": "could not parse arguments", "raw": raw_args})
        return await tools.execute(self.state, name, args)
