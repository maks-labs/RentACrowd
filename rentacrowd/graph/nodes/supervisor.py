"""The supervisor.

Every worker reports back here. The supervisor looks at what the study has so
far and decides who works next - or that the study is finished. It is an LLM
call, not a hardcoded sequence, so it can react to the request ("only the rural
segment", "reuse the panel from last time") instead of blindly marching through
a pipeline.

A deterministic guard runs first: if a prerequisite is genuinely missing there is
only one legal move, and we take it without spending a call.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from rentacrowd.events import emit
from rentacrowd.llm import analysis_llm, structured_call
from rentacrowd.state import StudyState

_NODE = "supervisor"

_ROSTER = """\
intake         - read the product brief, produce the test stimulus and the market segments
research       - scrape real customer language about the named competitors (reviews, forums)
recruit_panel  - assemble the persona panel: reuse people from the library, create new ones only if needed
run_panel      - fan out parallel workers so every persona reacts to the stimulus
moderator      - debrief the raw reactions: themes, tensions, questions a synthetic panel cannot settle
analyze        - compute the segment statistics and write the decision readout
FINISH         - the study is complete"""


class _Decision(BaseModel):
    next_action: str = Field(
        description="One of: intake, research, recruit_panel, run_panel, moderator, analyze, FINISH"
    )
    reason: str = Field(description="One short sentence: why this worker, now")


def _forced_move(state: StudyState) -> str | None:
    """The only legal move when a prerequisite is missing. Saves an LLM call."""
    done = state.get("completed", [])
    if not state.get("stimulus"):
        return "intake"
    if state.get("competitors") and "research" not in done:
        return "research"
    if not state.get("personas"):
        return "recruit_panel"
    if not state.get("responses"):
        return "run_panel"
    if state.get("report"):
        return "FINISH"
    return None


def supervisor(state: StudyState) -> dict:
    forced = _forced_move(state)
    if forced:
        emit(_NODE, "route", f"→ {forced}", reason="prerequisite missing", next_action=forced)
        return {"next_action": forced, "supervisor_log": [f"{forced} (required)"]}

    done = state.get("completed", [])
    status = (
        f"Product: {state['stimulus'].name} ({state['stimulus'].category})\n"
        f"Segments: {', '.join(s.name for s in state.get('segments', []))}\n"
        f"Panel: {len(state.get('personas', []))} personas — {state.get('recruitment_note', 'n/a')}\n"
        f"Reactions collected: {len(state.get('responses', []))}\n"
        f"Moderator debrief: {'yes' if state.get('moderator_notes') else 'no'}\n"
        f"Final report: {'yes' if state.get('report') else 'no'}\n"
        f"Workers already run: {', '.join(done) or 'none'}\n"
        f"User's extra instructions: {state.get('request_notes') or 'none'}"
    )

    emit(_NODE, "llm", "Deciding which worker runs next")
    decision = structured_call(
        analysis_llm(),
        _Decision,
        "You are the supervisor of a synthetic market-research study. Choose the "
        "single next worker to run. Never repeat a worker that has already run "
        "unless the user's instructions clearly ask for it.\n\n"
        f"WORKERS\n{_ROSTER}\n\n"
        f"STUDY SO FAR\n{status}",
    )

    action = decision.next_action.strip()
    if action not in {"intake", "research", "recruit_panel", "run_panel", "moderator", "analyze", "FINISH"}:
        action = "FINISH"

    emit(_NODE, "route", f"→ {action} — {decision.reason}", next_action=action)
    return {"next_action": action, "supervisor_log": [f"{action}: {decision.reason}"]}
