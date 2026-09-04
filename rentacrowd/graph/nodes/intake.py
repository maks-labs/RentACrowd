"""Intake node: free-text product blurb -> structured Stimulus + SegmentSpecs.

For now this is a single structured LLM call. In Phase 2 the body of this node
is replaced by a `deepagents` subgraph that also researches the category,
competitors and pricing norms before emitting the stimulus.
"""

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
    segments: list[SegmentSpec] = Field(description="3-4 segments with distinct reactions")


def intake(state: StudyState) -> dict:
    s = get_settings()
    emit(_NODE, "start", "Reading the product brief")
    emit(_NODE, "llm", f"Structuring it into a test stimulus + segments ({s.analysis_model})")
    out = structured_call(
        analysis_llm(),
        _Intake,
        "You are a market-research lead. Turn this rough product description into a "
        "precise test stimulus and 3-4 target market segments that would plausibly "
        "react differently to it.\n\n"
        f"Product description:\n{state['raw_product']}\n\n"
        f"Set each segment's `size` to {s.personas_per_segment}.",
    )
    # enforce configured panel size regardless of what the model chose
    for seg in out.segments:
        seg.size = s.personas_per_segment

    emit(
        _NODE,
        "result",
        f"Stimulus: {out.stimulus.name} @ {out.stimulus.price} · "
        f"{len(out.segments)} segments: {', '.join(seg.name for seg in out.segments)}",
        stimulus=out.stimulus.model_dump(),
        segments=[seg.model_dump() for seg in out.segments],
    )
    return {"stimulus": out.stimulus, "segments": out.segments}
