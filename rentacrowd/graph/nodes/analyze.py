"""Analysis node.

Numbers are computed in Python (deterministic, auditable). The LLM only writes
the narrative - headline, ranked objections, recommended changes, caveats - and
its numeric fields are then overwritten with the computed truth.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

from rentacrowd.config import get_settings
from rentacrowd.events import emit
from rentacrowd.llm import analysis_llm, structured_call
from rentacrowd.schemas import SegmentResult, StudyReport
from rentacrowd.state import StudyState

_NODE = "analyze"


def _pct_positive(sentiments: list[str]) -> float:
    if not sentiments:
        return 0.0
    return 100.0 * sum(s == "positive" for s in sentiments) / len(sentiments)


def analyze(state: StudyState) -> dict:
    responses = state["responses"]
    seg_by_id = {p.persona_id: p.segment for p in state["personas"]}

    by_seg: dict[str, list] = defaultdict(list)
    for r in responses:
        by_seg[seg_by_id.get(r.persona_id, "unknown")].append(r)

    segment_results = [
        SegmentResult(
            segment=seg,
            n=len(rs),
            mean_purchase_intent=round(mean(r.purchase_intent for r in rs), 2),
            pct_positive=round(_pct_positive([r.sentiment for r in rs]), 1),
            top_objections=_top([r.top_objection for r in rs]),
            wtp_summary="; ".join(sorted({r.willingness_to_pay for r in rs})[:5]),
        )
        for seg, rs in sorted(by_seg.items())
    ]

    overall_intent = round(mean(r.purchase_intent for r in responses), 2)
    overall_pos = round(_pct_positive([r.sentiment for r in responses]), 1)

    emit(
        _NODE,
        "compute",
        f"Aggregated {len(responses)} responses into {len(segment_results)} segment "
        f"stats in Python (deterministic) — overall intent {overall_intent}/5, "
        f"{overall_pos}% positive",
        segment_results=[s.model_dump() for s in segment_results],
    )
    emit(_NODE, "llm", f"Writing the decision-ready readout ({get_settings().analysis_model})")
    draft = structured_call(
        analysis_llm(),
        StudyReport,
        "Write a decision-ready synthetic-research readout. Be blunt about weak "
        "demand. Base every claim on the data provided.\n\n"
        f"Product: {state['stimulus'].name} @ {state['stimulus'].price}\n"
        f"Overall mean purchase intent: {overall_intent}/5\n"
        f"Overall % positive sentiment: {overall_pos}\n"
        f"Per-segment: {[s.model_dump() for s in segment_results]}\n"
        f"Moderator debrief: {state.get('moderator_notes', '')}\n"
        f"Open probes: {state.get('moderator_probes', [])}",
    )

    # overwrite model's numeric guesses with computed truth
    draft.overall_mean_purchase_intent = overall_intent
    draft.overall_pct_positive = overall_pos
    draft.segment_results = segment_results

    emit(_NODE, "result", f"Verdict: {draft.headline}", report=draft.model_dump())
    return {"report": draft}


def _top(items: list[str], k: int = 3) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for it in items:
        counts[it.strip().lower()] += 1
    return [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:k]]
