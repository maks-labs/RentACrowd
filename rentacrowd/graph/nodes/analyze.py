"""Analysis worker.

Numbers are computed in Python (deterministic, auditable). The LLM writes only
the narrative - headline, ranked objections, recommended changes, caveats - and
its numeric fields are overwritten with the computed truth. Then the whole study
is written to a folder you can open.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
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


def _top(items: list[str], k: int = 3) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for it in items:
        counts[it.strip().lower()] += 1
    return [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:k]]


def _write_study(state: StudyState, report: StudyReport) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = "".join(c if c.isalnum() else "-" for c in state["stimulus"].name.lower())[:30].strip("-")
    d = get_settings().studies_dir / f"{stamp}--{slug or 'study'}"
    d.mkdir(parents=True, exist_ok=True)

    (d / "stimulus.json").write_text(json.dumps(state["stimulus"].model_dump(), indent=2))
    if state.get("market_evidence"):
        (d / "market_evidence.json").write_text(
            json.dumps(state["market_evidence"], indent=2)
        )
    (d / "panel.json").write_text(
        json.dumps([p.model_dump() for p in state["personas"]], indent=2)
    )
    (d / "responses.json").write_text(
        json.dumps([r.model_dump() for r in state["responses"]], indent=2)
    )
    (d / "report.json").write_text(json.dumps(report.model_dump(), indent=2))
    (d / "report.md").write_text(_report_md(state, report))
    return str(d)


def _report_md(state: StudyState, r: StudyReport) -> str:
    L = [
        f"# {state['stimulus'].name} — synthetic study",
        "",
        f"**{r.headline}**",
        "",
        f"- Mean purchase intent: **{r.overall_mean_purchase_intent} / 5**",
        f"- Positive sentiment: **{r.overall_pct_positive}%**",
        f"- Panel: {len(state['personas'])} personas — {state.get('recruitment_note','')}",
        "",
        "## By segment",
        "",
        "| segment | n | intent | % positive | top objections |",
        "| --- | --- | --- | --- | --- |",
    ]
    L += [
        f"| {s.segment} | {s.n} | {s.mean_purchase_intent} | {s.pct_positive} | "
        f"{'; '.join(s.top_objections)} |"
        for s in r.segment_results
    ]
    L += ["", "## Key objections", ""] + [f"- {o}" for o in r.key_objections]
    L += ["", "## Recommended changes", ""] + [f"- {c}" for c in r.recommended_changes]
    L += ["", "## Moderator debrief", "", state.get("moderator_notes", "")]
    L += ["", "## Probes for a real validation panel", ""]
    L += [f"- {p}" for p in state.get("moderator_probes", [])]
    L += ["", f"> Confidence — {r.confidence_notes}", ""]
    return "\n".join(L)


def analyze(state: StudyState) -> dict:
    responses = state["responses"]
    personas = state["personas"]
    seg_by_id = {p.persona_id.lower(): p.segment for p in personas}

    def _seg_for(rid: str) -> str:
        key = rid.lower()
        if key in seg_by_id:
            return seg_by_id[key]
        # tolerate the model abbreviating an id: match on prefix
        for pid, seg in seg_by_id.items():
            if pid.startswith(key) or key.startswith(pid):
                return seg
        return personas[0].segment if personas else "unknown"

    by_seg: dict[str, list] = defaultdict(list)
    for r in responses:
        by_seg[_seg_for(r.persona_id)].append(r)

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
        f"Aggregated {len(responses)} reactions → {len(segment_results)} segment stats "
        f"in Python — intent {overall_intent}/5, {overall_pos}% positive",
        segment_results=[s.model_dump() for s in segment_results],
    )
    emit(_NODE, "llm", f"Writing the decision readout ({get_settings().analysis_model})")
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
    draft.overall_mean_purchase_intent = overall_intent
    draft.overall_pct_positive = overall_pos
    draft.segment_results = segment_results

    study_dir = _write_study(state, draft)
    emit(_NODE, "result", f"Verdict: {draft.headline}", report=draft.model_dump(), study_dir=study_dir)
    return {"report": draft, "study_dir": study_dir, "completed": [_NODE]}
