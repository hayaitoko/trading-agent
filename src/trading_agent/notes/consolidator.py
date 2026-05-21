"""Memory consolidator. Curates notes without deleting information.

Design tenet (set by the user): provenance over freshness. Old observations
stay; they just need to be dated so the agent knows what's archival vs current.

What this consolidator does:
- Adds missing frontmatter (title, created, updated) to any note without it.
- Adds inline `(as of YYYY-MM-DD)` markers to time-bound claims that lack them.
- Merges notes that are structurally about the same subject (two NVDA notes).
- Improves formatting and cross-links related notes.

What it does NOT do:
- Delete information.
- Mark things stale because they are old.
- Make any change without a `.history/<timestamp>/` backup of the original.
"""
import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from trading_agent.chat.client import OpenRouterError, call_model
from trading_agent.chat.models import DEFAULT_MODEL, ChatMessage, find_model
from trading_agent.notes import storage

ModelCaller = Callable[..., Awaitable[dict[str, Any]]]
ApiKeyGetter = Callable[[], str]


@dataclass
class ConsolidatorConfig:
    enabled: bool = False
    interval_minutes: int = 60
    model: str = DEFAULT_MODEL


@dataclass
class ConsolidatorStatus:
    last_run_at: str | None = None
    last_run_summary: str | None = None
    last_error: str | None = None
    next_run_at: str | None = None
    running: bool = False
    edits_last_run: int = 0


@dataclass
class _RunResult:
    summary: str
    edits: list[dict] = field(default_factory=list)
    error: str | None = None


SYSTEM_PROMPT = (
    "You are the memory consolidator for an autonomous trading agent. Your job "
    "is to curate the agent's notes (markdown files) without losing information.\n\n"
    "CORE RULE: provenance over freshness. Old observations stay; they just need "
    "to be dated so the agent knows what is archival vs current. You DO NOT have "
    "access to live news or market data. You CANNOT determine whether a claim is "
    "currently true. You only ever make a note safer to read across time.\n\n"
    "Operations you may perform:\n"
    "1. Add missing frontmatter (`title`, `created`, `updated`) to notes that "
    "lack it. Use today's date for `updated`; use a best-effort guess for "
    "`created` (or today if unknown).\n"
    "2. Add inline `(as of YYYY-MM-DD)` markers to time-bound claims that have "
    "none. Use today's date when no other signal is available.\n"
    "3. Merge two notes that are structurally about the same subject into one "
    "(same ticker, same sector, same topic). Preserve every observation from "
    "both; just consolidate the structure. When merging, the surviving file's "
    "edit is the merged content; the absorbed file's edit is an empty string to "
    "indicate deletion (the storage layer will keep a backup).\n"
    "4. Improve formatting: consistent headings, bullet structure, cross-links.\n\n"
    "Operations forbidden:\n"
    "- Deleting any observation, even if it sounds stale.\n"
    "- Removing dated entries because the date is old.\n"
    "- Rewriting claims to change their meaning.\n"
    "- Adding facts you cannot derive from the notes themselves.\n\n"
    "Output a single JSON object:\n"
    "{\n"
    '  "summary": "one short paragraph describing what you changed",\n'
    '  "edits": [\n'
    '    {"path": "companies/NVDA.md", "new_content": "<full file content>"}\n'
    "  ]\n"
    "}\n"
    "Include ONLY paths you want to change. Omit notes that need no changes. "
    "An edit with empty `new_content` means delete (used for merge-absorption only)."
)


class Consolidator:
    def __init__(
        self,
        *,
        notes_dir: Path,
        config_path: Path,
        log_path: Path,
        api_key_getter: ApiKeyGetter,
        model_caller: ModelCaller | None = None,
    ):
        self.notes_dir = notes_dir
        self.config_path = config_path
        self.log_path = log_path
        self._get_api_key = api_key_getter
        self._model_caller = model_caller or call_model
        self.status = ConsolidatorStatus()
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    def load_config(self) -> ConsolidatorConfig:
        if not self.config_path.exists():
            return ConsolidatorConfig()
        try:
            raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return ConsolidatorConfig()
        return ConsolidatorConfig(
            enabled=bool(raw.get("enabled", False)),
            interval_minutes=max(1, int(raw.get("interval_minutes", 60))),
            model=str(raw.get("model", DEFAULT_MODEL)),
        )

    def save_config(self, config: ConsolidatorConfig) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            json.dumps(asdict(config), indent=2), encoding="utf-8"
        )
        self._update_next_run(config)

    def _update_next_run(self, config: ConsolidatorConfig) -> None:
        if not config.enabled:
            self.status.next_run_at = None
            return
        base = (
            datetime.fromisoformat(self.status.last_run_at)
            if self.status.last_run_at
            else datetime.now()
        )
        self.status.next_run_at = (
            base + timedelta(minutes=config.interval_minutes)
        ).isoformat(timespec="seconds")

    async def run_once(self) -> _RunResult:
        async with self._lock:
            self.status.running = True
            self.status.last_error = None
            try:
                config = self.load_config()
                notes = self._load_all_notes()
                if not notes:
                    return _RunResult(summary="no notes to consolidate")
                model_spec = find_model(config.model) or find_model(DEFAULT_MODEL)
                if model_spec is None:
                    raise OpenRouterError(f"unknown model: {config.model}")
                api_key = self._get_api_key()
                response = await self._model_caller(
                    api_key=api_key,
                    model=model_spec,
                    system_prompt=SYSTEM_PROMPT,
                    history=[ChatMessage(
                        role="user",
                        content=self._build_user_message(notes),
                    )],
                    tools=[],
                )
                result = self._parse_response(response.get("content") or "")
                self._apply_edits(result.edits)
                self._record(result, config, model_spec.id)
                return result
            except Exception as e:
                self.status.last_error = str(e)
                self._append_log(
                    f"ERROR at {datetime.now().isoformat(timespec='seconds')}: {e}\n"
                )
                return _RunResult(summary="run failed", error=str(e))
            finally:
                self.status.running = False

    def _load_all_notes(self) -> list[tuple[str, str]]:
        notes: list[tuple[str, str]] = []
        for path in sorted(self.notes_dir.rglob(f"*{storage.NOTE_EXT}")):
            if any(part in storage.RESERVED_DIRS for part in path.relative_to(self.notes_dir).parts):
                continue
            rel = str(path.relative_to(self.notes_dir)).replace("\\", "/")
            notes.append((rel, path.read_text(encoding="utf-8")))
        return notes

    def _build_user_message(self, notes: list[tuple[str, str]]) -> str:
        parts = [f"Today's date: {datetime.now().date().isoformat()}", ""]
        parts.append(f"There are {len(notes)} notes to review:\n")
        for rel, content in notes:
            parts.append(f"<note path=\"{rel}\">")
            parts.append(content.rstrip())
            parts.append("</note>")
            parts.append("")
        parts.append(
            "Review them per the rules. Return only the JSON object with "
            "your proposed edits."
        )
        return "\n".join(parts)

    def _parse_response(self, raw: str) -> _RunResult:
        # Model may wrap the JSON in a ```json``` fence; strip it.
        cleaned = re.sub(r"^```(?:json)?\n?|\n?```$", "", raw.strip(), flags=re.MULTILINE)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise OpenRouterError(f"could not parse JSON: {e}; raw={raw[:200]}") from e
        edits = data.get("edits") or []
        if not isinstance(edits, list):
            raise OpenRouterError("edits is not a list")
        return _RunResult(
            summary=str(data.get("summary") or "(no summary)"),
            edits=[
                {"path": str(e.get("path", "")), "new_content": str(e.get("new_content", ""))}
                for e in edits
            ],
        )

    def _apply_edits(self, edits: list[dict]) -> None:
        for edit in edits:
            path = edit["path"]
            new_content = edit["new_content"]
            if not new_content:
                # Merge-absorption: delete original. delete_note snapshots first.
                try:
                    storage.delete_note(self.notes_dir, path)
                except storage.NotesError:
                    pass
            else:
                storage.write_note(self.notes_dir, path, new_content)

    def _record(self, result: _RunResult, config: ConsolidatorConfig, model_id: str) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        self.status.last_run_at = now
        self.status.last_run_summary = result.summary
        self.status.edits_last_run = len(result.edits)
        self._update_next_run(config)
        log_block = (
            f"## {now} ({model_id})\n\n"
            f"{result.summary}\n\n"
            f"Edits: {len(result.edits)}\n"
        )
        for edit in result.edits:
            label = "DELETE" if not edit["new_content"] else "WRITE"
            log_block += f"- {label} `{edit['path']}`\n"
        log_block += "\n"
        self._append_log(log_block)

    def _append_log(self, text: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(text)

    def start_loop(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def stop_loop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _loop(self) -> None:
        # Wakes every minute. Only invokes run_once when enabled and due.
        # Bounded retry on transient failures via the run_once try/except.
        while True:
            try:
                config = self.load_config()
                if config.enabled and self._is_due(config):
                    await self.run_once()
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.status.last_error = f"loop: {e}"
                await asyncio.sleep(60)

    def _is_due(self, config: ConsolidatorConfig) -> bool:
        if not self.status.last_run_at:
            return True
        last = datetime.fromisoformat(self.status.last_run_at)
        return datetime.now() - last >= timedelta(minutes=config.interval_minutes)
