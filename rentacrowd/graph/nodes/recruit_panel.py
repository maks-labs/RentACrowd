"""Recruit the panel.

Recruiting is not generating. This worker first tries to staff every segment
from the existing persona library - real files on disk that persist between
studies - and only manufactures new people for the shortfall, or when the user
explicitly asks for fresh ones. New hires are written back to the library, so
the population compounds instead of being thrown away after each study.
"""

from __future__ import annotations

import re
from collections import Counter

from rentacrowd.config import get_settings
from rentacrowd.events import emit
from rentacrowd.personas import library
from rentacrowd.personas.generate import make_personas
from rentacrowd.schemas import Persona
from rentacrowd.state import StudyState

_NODE = "recruit_panel"

_FRESH = re.compile(r"\b(fresh|new personas|regenerate|different people|new panel)\b", re.I)
_STOP = set(
    "a an the and or of to in for with on at by who are is who's their they them "
    "people person who that this those these households household who prefer prefer "
    "want new use using used over more less than not no every each any some most "
    "into onto from as it its be been being do does did has have had".split()
)


def _words(text: str) -> Counter:
    toks = re.findall(r"[a-z][a-z'-]{2,}", text.lower())
    return Counter(t for t in toks if t not in _STOP)


def _persona_text(p: Persona) -> str:
    # tags and life-stage words repeated so they weigh more in the overlap score
    return " ".join(
        [p.occupation, p.household, p.household, p.income_band, p.psychographics,
         p.current_solutions, p.frustration, p.decision_style,
         p.price_sensitivity, p.tech_comfort, *p.tags, *p.tags, *p.tags]
    )


def _cast_from_library(state: StudyState, per_segment: int) -> dict[str, list[str]]:
    """Deterministic keyword-overlap casting - fast, no LLM, reliable reuse.

    Each library persona is scored against each segment by shared vocabulary
    between the segment's definition and the persona's own words; the best
    unclaimed fits fill the segment.
    """
    people = library.load_all()
    if not people:
        return {}

    pvecs = [(p, _words(_persona_text(p))) for p in people]
    claimed: set[str] = set()
    out: dict[str, list[str]] = {}

    for seg in state["segments"]:
        svec = _words(f"{seg.name} {seg.description}")
        scored = sorted(
            (
                (sum(min(svec[w], pv[w]) for w in svec), p.persona_id)
                for p, pv in pvecs
                if p.persona_id not in claimed
            ),
            reverse=True,
        )
        picks = [pid for score, pid in scored if score >= 2][:per_segment]
        claimed.update(picks)
        out[seg.name] = picks
    return out


def recruit_panel(state: StudyState) -> dict:
    s = get_settings()
    per_segment = state.get("panel_size") or s.personas_per_segment
    segments = state["segments"]
    wants_fresh = bool(_FRESH.search(state.get("request_notes") or ""))

    emit(
        _NODE,
        "start",
        f"Staffing {len(segments)} segments × {per_segment} people "
        f"(library holds {library.size()})",
    )

    cast: dict[str, list[str]] = {}
    if wants_fresh:
        emit(_NODE, "cache", "User asked for fresh personas — skipping the library")
    elif library.size() == 0:
        emit(_NODE, "cache", "Library is empty — everyone will be newly created")
    else:
        emit(_NODE, "compute", f"Matching {library.size()} library personas to the segments")
        cast = _cast_from_library(state, per_segment)

    panel: list[Persona] = []
    reused = created = 0

    for seg in segments:
        chosen_ids = cast.get(seg.name, [])
        chosen = library.load_by_ids(chosen_ids)
        for p in chosen:
            p.segment = seg.name          # relabel for this study; identity unchanged
        if chosen:
            emit(
                _NODE,
                "cache",
                f"REUSED {len(chosen)} from library for '{seg.name}': "
                + ", ".join(p.name for p in chosen),
            )
        panel.extend(chosen)
        reused += len(chosen)

        shortfall = per_segment - len(chosen)
        if shortfall > 0:
            emit(
                _NODE,
                "spawn",
                f"persona-generator → {shortfall} new for '{seg.name}' "
                f"(library had no fit) ({s.panel_model})",
            )
            fresh = library.add(make_personas(seg.name, seg.description, shortfall))
            panel.extend(fresh)
            created += len(fresh)

    note = (
        f"{reused} reused from the library, {created} newly created "
        f"(library now holds {library.size()})"
    )
    emit(_NODE, "result", f"Panel of {len(panel)}: {note}", personas=[p.model_dump() for p in panel])
    return {"personas": panel, "recruitment_note": note, "completed": [_NODE]}
