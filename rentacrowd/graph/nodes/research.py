"""Research worker.

Before the panel reacts, pull real customer language about the named competitors
- App Store reviews, Hacker News (and Reddit if OAuth is configured) - and
distil it into evidence the panel workers can echo, so simulated objections and
vocabulary track what real people actually say.

Best-effort: competitors with no findable footprint are noted and skipped.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

from rentacrowd.config import get_settings
from rentacrowd.events import emit
from rentacrowd.llm import analysis_llm, structured_call
from rentacrowd.research import scrape
from rentacrowd.state import StudyState

_NODE = "research"


class MarketEvidence(BaseModel):
    competitor: str
    real_objections: list[str] = Field(
        description="Complaints real users actually make, phrased the way they phrase them"
    )
    praised: list[str] = Field(description="What real users like about it")
    authentic_phrases: list[str] = Field(
        description="Short verbatim-style phrases / vocabulary real users use for this category"
    )
    unmet_needs: list[str] = Field(description="Gaps users mention that nothing solves well")


def _distil(bundle: dict) -> MarketEvidence | None:
    # negative & mixed reviews carry the signal; skip content-free 5-star blurbs
    reviews = [r for r in bundle["reviews"] if r["rating"] <= 4 or len(r["text"]) > 120][:45]
    disc = bundle["discussion"][:18]
    if not reviews and not disc:
        return None
    corpus = "\n".join(
        [f"[{r['rating']}star] {r['text']}" for r in reviews]
        + [f"[forum] {d['text']}" for d in disc]
    )[:7000]
    return structured_call(
        analysis_llm(),
        MarketEvidence,
        f"Real user feedback about {bundle['competitor']} (a competitor in this "
        f"market). Extract what real customers actually say - keep their wording. "
        f"Ignore spam. Keep every list to at most 6 short items.\n\n"
        f"FEEDBACK\n{corpus}\n\n"
        f'Set `competitor` to "{bundle["competitor"]}".',
    )


def research(state: StudyState) -> dict:
    competitors = [c for c in (state.get("competitors") or []) if c.strip()][:3]
    if not competitors:
        emit(_NODE, "result", "No competitors named — skipping external research")
        return {"market_evidence": [], "completed": [_NODE]}

    emit(_NODE, "start", f"Gathering real customer language on: {', '.join(competitors)}")

    with ThreadPoolExecutor(max_workers=3) as pool:
        bundles = list(pool.map(scrape.gather, competitors))

    evidence: list[dict] = []
    for b in bundles:
        c = b["counts"]
        emit(
            _NODE,
            "spawn",
            f"{b['competitor']}: {c['itunes_reviews']} App Store reviews, "
            f"{c['discussion_posts']} forum posts"
            + (f" (matched app: {b['app_match']['name']})" if b["app_match"] else " (no app match)"),
        )
        if c["itunes_reviews"] == 0 and c["discussion_posts"] == 0:
            continue
        emit(_NODE, "llm", f"Distilling {b['competitor']} feedback into evidence ({get_settings().analysis_model})")
        ev = _distil(b)
        if ev:
            evidence.append(ev.model_dump() | {"counts": c})

    if evidence:
        n_obj = sum(len(e["real_objections"]) for e in evidence)
        emit(
            _NODE,
            "result",
            f"{len(evidence)} competitor(s) analysed — {n_obj} real objections captured",
            evidence=evidence,
        )
    else:
        emit(_NODE, "result", "No usable public feedback found — proceeding on personas alone")
    return {"market_evidence": evidence, "completed": [_NODE]}


def evidence_block(evidence: list[dict]) -> str:
    """Render evidence for the panel-worker prompt. Empty string if there is none."""
    if not evidence:
        return ""
    parts = ["REAL CUSTOMER EVIDENCE (from reviews & forums about competitors):"]
    for e in evidence:
        parts.append(
            f"\n• {e['competitor']}\n"
            f"  objections people actually raise: {'; '.join(e['real_objections'][:6])}\n"
            f"  what they praise: {'; '.join(e['praised'][:4])}\n"
            f"  phrases they use: {'; '.join(e['authentic_phrases'][:8])}\n"
            f"  unmet needs: {'; '.join(e['unmet_needs'][:4])}"
        )
    parts.append(
        "\nWhere a persona's situation fits, ground their objection and wording in this."
    )
    return "\n".join(parts)
