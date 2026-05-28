"""Tutorial prompt templates for new AgentTrader instances (WS-Agent A6).

The first ``tutorial_remaining`` turns a new trader takes use guided extra_lines
injected into the always-on first-look block.  Each tutorial turn has one focus —
turn 1 introduces the tool catalog, turn 2 demonstrates memory, turn 3 demonstrates
watchpoints.  Beyond turn 3, a generic "keep exploring" message is returned for any
custom ``tutorial_remaining`` configured above 3.

After ``tutorial_remaining`` reaches 0 **or** after the first ``trade*`` terminal,
the trader operates freely on normal prompts — no tutorial guidance appears.

Design role:
  Called from ``AgentTrader._turn_type_guidance("tutorial")`` which passes
  ``(turn_num, total_turns)`` derived from ``self._tutorial_total`` and
  ``self.tutorial_remaining``.  The return value is assigned to
  ``TurnContext.extra_lines`` and rendered verbatim into the first-look block.

Failure mode:
  None.  ``tutorial_extra_lines`` is pure string construction; it cannot fail.
  The caller always gets a non-empty list with at least the header line.

**MONEY IS REAL invariant:** these strings are trader-facing.  The words
"paper", "sim", "demo", "fake", and "test mode" must never appear here.
See design/TRADER-AGENT.md §MONEY IS REAL.
"""


def tutorial_extra_lines(turn_number: int, total_turns: int) -> list[str]:
    """Return extra_lines for tutorial turn ``turn_number`` of ``total_turns``.

    Args:
        turn_number: 1-based index of the current tutorial turn (1 = first).
        total_turns: total tutorial turns this trader was configured with.

    Returns:
        List of strings for ``TurnContext.extra_lines``.  Starts with ``""``
        so the tutorial block is visually separated from the attention counts.
    """
    remaining_after = total_turns - turn_number
    if remaining_after == 0:
        suffix = "after this turn you decide freely — no more guided prompts."
    elif remaining_after == 1:
        suffix = "1 guided turn remains after this one."
    else:
        suffix = f"{remaining_after} guided turns remain after this one."

    header = f"\nTutorial — turn {turn_number} of {total_turns}: {suffix}"

    if turn_number == 1:
        return [
            header,
            "STEP 1 — Discover your tools.  Call list_tools() to see the full "
            "catalog: LOOK tools read data and context, NOTE tools record lessons "
            "and set monitors, ACT tools submit orders, and END terminals "
            "(hold / pass / done_for_day) close the turn.  "
            "After reviewing the catalog, call pass() or hold() to end this turn.",
        ]

    if turn_number == 2:
        return [
            header,
            "STEP 2 — Explore memory.  Call memory_search(query='first session') "
            "to inspect your reflection history — it will be empty since you are "
            "new here.  Then call reflect(note='<a lesson you want to keep>') to "
            "write your first durable memory.  Finish with hold() or pass().",
        ]

    if turn_number == 3:
        return [
            header,
            "STEP 3 — Set a watchpoint.  Call watchpoint(symbol=<a symbol from "
            "your universe>, why='watching for my first opportunity') to register "
            "a standing monitor.  When that symbol moves interestingly you will "
            "receive an event turn.  Finish with hold() or pass().",
        ]

    # Configurable tutorial_remaining > 3 — generic "keep exploring" message.
    return [
        header,
        "Use this turn to try any tools you have not yet explored.  "
        "Finish with hold() or pass() when you are done.",
    ]
