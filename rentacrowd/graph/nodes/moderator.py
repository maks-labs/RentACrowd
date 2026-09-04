"""Moderator worker.

Reads the whole panel's raw reactions and does the human moderator's after-job:
name the recurring themes, where segments split, and the questions a synthetic
panel genuinely cannot answer. One LLM call.
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
    probes: list[str] = Field(description="3-5 questions to put to a real validation panel")


def moderator(state: StudyState) -> dict:
    responses = state["responses"]
    sample = responses[:80]
    emit(_NODE, "start", f"Moderator agent debriefing {len(responses)} reactions")

    lines = "\n".join(
        f"- {r.persona_id}: intent {r.purchase_intent}/5 ({r.sentiment}); "
        f'objection: {r.top_objection}; switching cost: {r.switching_cost}; "{r.verbatim}"'
        for r in sample
    )
    emit(_NODE, "llm", f"Summarising themes + drafting validation probes ({get_settings().analysis_model})")
    out = structured_call(
        analysis_llm(),
        _Debrief,
        "You are a senior qualitative research moderator debriefing after a panel. "
        "Identify the real signal: recurring themes, where segments diverge, and the "
        "open questions a synthetic panel cannot settle.\n\n"
        f"Product: {state['stimulus'].name} — {state['stimulus'].description}\n\n"
        f"Reactions ({len(sample)} of {len(responses)}):\n{lines}",
    )
    emit(
        _NODE,
        "result",
        f"{len(out.probes)} probe questions for a real validation panel",
        notes=out.notes,
        probes=out.probes,
    )
    return {
        "moderator_notes": out.notes,
        "moderator_probes": out.probes,
        "completed": [_NODE],
    }
