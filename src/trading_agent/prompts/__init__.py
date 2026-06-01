"""Prompt templates package.

Provides structured prompt components used by the agent trader loop.
Currently contains tutorial-mode guidance (A6) and the trader persona
registry (``personas.py``); future phases may add SoD/EoD structured
prompts here as they evolve beyond simple extra_lines.
"""

from .personas import PERSONAS, PERSONAS_BY_ID, PersonaEntry, get_persona_mandate

__all__ = [
    "PERSONAS",
    "PERSONAS_BY_ID",
    "PersonaEntry",
    "get_persona_mandate",
]
