"""Markdown-file note storage with path validation, history backups, and search.

Conventions enforced by the consolidator layer (not here):
- Every note has frontmatter with `created` and `updated` ISO dates.
- Time-bound claims inside the body should carry inline `(as of YYYY-MM-DD)` markers.

This storage module is intentionally dumb about content — it just reads and
writes UTF-8 markdown files. The consolidator adds the conventions over time.
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

NOTE_EXT = ".md"

# Folders auto-created on first use. The consolidator and the chat-side tools
# can write into any of these. User-created subfolders also work; the tree
# walker discovers them.
DEFAULT_NOTE_DIRS: tuple[str, ...] = ("companies", "sectors", "macro", "general")

# Reserved folder names the user/LLM may not write into directly.
RESERVED_DIRS: frozenset[str] = frozenset({".history", ".consolidator"})

DEFAULT_README = """# Notes

This is the agent's memory. Organized by intent:

- `companies/` — per-ticker research and observations
- `sectors/` — industry trends (tech, energy, finance, etc.)
- `macro/` — Fed policy, inflation, geopolitics, anything market-wide
- `general/` — anything that doesn't fit above

## Convention: timestamp everything

Every note has frontmatter with `created` and `updated` dates. Time-bound
claims inside a note carry inline `(as of YYYY-MM-DD)` markers. The
consolidator backfills missing timestamps automatically.

Old observations stay. The point of memory is provenance — knowing what
we thought when, not just what we think now.

## Memory consolidator

The consolidator (configurable on this page) runs on a timer to merge
structurally duplicate notes, add missing timestamps, and improve formatting.
It does not delete information. Original versions are kept under `.history/`
in case you want to revert.
"""


class NotesError(Exception):
    pass


@dataclass
class NoteNode:
    name: str
    path: str
    type: str  # "file" or "dir"
    children: list["NoteNode"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "type": self.type,
            "children": [c.to_dict() for c in self.children],
        }


def _safe_resolve(notes_dir: Path, rel_path: str) -> Path:
    """Resolve rel_path under notes_dir, blocking traversal and reserved dirs."""
    if not rel_path:
        raise NotesError("path is empty")
    rel = Path(rel_path)
    if rel.is_absolute():
        raise NotesError("path must be relative")
    parts = rel.parts
    if any(p in ("..", "") for p in parts):
        raise NotesError("path contains invalid segments")
    if parts and parts[0] in RESERVED_DIRS:
        raise NotesError(f"path is reserved: {parts[0]}")
    target = (notes_dir / rel).resolve()
    try:
        target.relative_to(notes_dir.resolve())
    except ValueError as e:
        raise NotesError("path escapes notes directory") from e
    return target


def _safe_file_path(notes_dir: Path, rel_path: str) -> Path:
    target = _safe_resolve(notes_dir, rel_path)
    if target.suffix != NOTE_EXT:
        raise NotesError(f"note path must end with {NOTE_EXT}")
    return target


def ensure_default_structure(notes_dir: Path) -> None:
    notes_dir.mkdir(parents=True, exist_ok=True)
    for sub in DEFAULT_NOTE_DIRS:
        (notes_dir / sub).mkdir(exist_ok=True)
    readme = notes_dir / "general" / "README.md"
    if not readme.exists():
        readme.write_text(DEFAULT_README, encoding="utf-8")


def list_tree(notes_dir: Path) -> NoteNode:
    notes_dir.mkdir(parents=True, exist_ok=True)
    root = NoteNode(name=notes_dir.name, path="", type="dir")
    _walk(notes_dir, notes_dir, root)
    return root


def _walk(root_dir: Path, current: Path, parent: NoteNode) -> None:
    entries = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    for entry in entries:
        if entry.name.startswith(".") and entry.name in RESERVED_DIRS:
            continue
        if entry.name.startswith(".") and entry.is_dir():
            continue
        rel = str(entry.relative_to(root_dir)).replace("\\", "/")
        if entry.is_dir():
            node = NoteNode(name=entry.name, path=rel, type="dir")
            _walk(root_dir, entry, node)
            parent.children.append(node)
        elif entry.suffix == NOTE_EXT:
            parent.children.append(NoteNode(name=entry.name, path=rel, type="file"))


def read_note(notes_dir: Path, rel_path: str) -> str:
    target = _safe_file_path(notes_dir, rel_path)
    if not target.exists():
        raise NotesError(f"note not found: {rel_path}")
    return target.read_text(encoding="utf-8")


def write_note(
    notes_dir: Path,
    rel_path: str,
    content: str,
    *,
    snapshot: bool = True,
) -> None:
    target = _safe_file_path(notes_dir, rel_path)
    if snapshot and target.exists():
        _snapshot(notes_dir, rel_path, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def delete_note(notes_dir: Path, rel_path: str) -> None:
    target = _safe_file_path(notes_dir, rel_path)
    if not target.exists():
        raise NotesError(f"note not found: {rel_path}")
    _snapshot(notes_dir, rel_path, target)
    target.unlink()


def _snapshot(notes_dir: Path, rel_path: str, target: Path) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    history_path = notes_dir / ".history" / stamp / rel_path
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")


def search_notes(notes_dir: Path, query: str, *, limit: int = 20) -> list[dict]:
    """Substring search across all notes. Returns matches with brief context."""
    if not query.strip():
        return []
    needle = query.lower()
    hits: list[dict] = []
    for path in sorted(notes_dir.rglob(f"*{NOTE_EXT}")):
        if any(part in RESERVED_DIRS for part in path.relative_to(notes_dir).parts):
            continue
        text = path.read_text(encoding="utf-8")
        idx = text.lower().find(needle)
        if idx == -1:
            continue
        start = max(0, idx - 60)
        end = min(len(text), idx + len(needle) + 60)
        excerpt = text[start:end].replace("\n", " ")
        hits.append({
            "path": str(path.relative_to(notes_dir)).replace("\\", "/"),
            "excerpt": ("..." if start > 0 else "") + excerpt + ("..." if end < len(text) else ""),
        })
        if len(hits) >= limit:
            break
    return hits
