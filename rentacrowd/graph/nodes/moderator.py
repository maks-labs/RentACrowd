"""Moderator node.

Reads the whole panel's raw responses and does what a human focus-group
moderator does afterwards: names the recurring themes, the tensions, and the
follow-up questions worth putting to a real validation panel. One LLM call.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from rentacrowd.config import get_settings
from rentacrowd.events import emit
from rentacrowd.llm import analysis_llm, structured_call
from rentacrowd.state import StudyState

_NODE = "moderator"


class _Debrief(BaseModel):
    notes: str = Field(description="3-5 sentence qualitative debrief")
    probes: list[str] = Field(description="3-5 follow-up questions for a real validation panel")


def moderator(state: StudyState) -> dict:
    responses = state["responses"]
    sample = responses[:60]
    emit(
        _NODE,
        "start",
        f"Moderator agent debriefing {len(responses)} panel responses",
    )
    lines = "\n".join(
        f"- intent {r.purchase_intent}/5, {r.sentiment}; objection: {r.top_objection}; "
        f'"{r.verbatim}"'
        for r in sample
    )
    emit(_NODE, "llm", f"Summarising themes + drafting validation probes ({get_settings().analysis_model})")
    out = structured_call(
        analysis_llm(),
        _Debrief,
        "You are a senior qualitative research moderator debriefing after a panel. "
        "Identify the real signal: recurring themes, where segments diverge, and the "
        "open questions a synthetic panel cannot settle.\n\n"
        f"Product: {state['stimulus'].name} - {state['stimulus'].description}\n\n"
        f"Panel responses ({len(sample)} of {len(responses)} shown):\n{lines}",
    )
    emit(
        _NODE,
        "result",
        f"{len(out.probes)} probe questions for a real validation panel",
        notes=out.notes,
        probes=out.probes,
    )
    return {"moderator_notes": out.notes, "moderator_probes": out.probes}
