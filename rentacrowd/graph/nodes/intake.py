"""Intake worker: free-text product brief -> structured Stimulus + SegmentSpecs."""

from __future__ import annotations

from pydantic import BaseModel, Field

from rentacrowd.config import get_settings
from rentacrowd.events import emit
from rentacrowd.llm import analysis_llm, structured_call
from rentacrowd.schemas import SegmentSpec, Stimulus
from rentacrowd.state import StudyState

_NODE = "intake"


class _Intake(BaseModel):
    stimulus: Stimulus
    segments: list[SegmentSpec] = Field(description="exactly 4 segments that would react differently")


def intake(state: StudyState) -> dict:
    s = get_settings()
    per_segment = state.get("panel_size") or s.personas_per_segment

    emit(_NODE, "start", "Reading the product brief")
    emit(_NODE, "llm", f"Structuring the stimulus + market segments ({s.analysis_model})")

    notes = state.get("request_notes")
    out = structured_call(
        analysis_llm(),
        _Intake,
        "You are a market-research lead. Turn this rough brief into a precise test "
        "stimulus and EXACTLY 4 target market segments that would plausibly react "
        "differently to the product. Segments must be about the people (life stage, "
        "budget, values, context), not about product tiers.\n\n"
        f"BRIEF\n{state['raw_product']}\n\n"
        + (f"The requester also said: {notes}\n\n" if notes else "")
        + f"Set every segment's `size` to {per_segment}.",
    )
    for seg in out.segments:
        seg.size = per_segment

    emit(
        _NODE,
        "result",
        f"{out.stimulus.name} @ {out.stimulus.price} · "
        f"{len(out.segments)} segments: {', '.join(seg.name for seg in out.segments)}",
        stimulus=out.stimulus.model_dump(),
        segments=[seg.model_dump() for seg in out.segments],
    )
    return {
        "stimulus": out.stimulus,
        "segments": out.segments,
        "completed": [_NODE],
    }
