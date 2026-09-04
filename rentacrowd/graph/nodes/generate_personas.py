"""Persona generation node.

Personas are generated in chunks of `batch_size` per segment (one LLM call each)
to keep any single request small and fast. Results are cached to disk keyed by
(category, segment-set); a cache hit costs zero LLM calls.
"""

from __future__ import annotations

from rentacrowd.config import get_settings
from rentacrowd.events import emit
from rentacrowd.llm import panel_llm, structured_call
from rentacrowd.personas import store
from rentacrowd.schemas import Persona, PersonaBatch, SegmentSpec, Stimulus
from rentacrowd.state import StudyState

_NODE = "generate_personas"


def _generate_chunk(
    stimulus: Stimulus, seg: SegmentSpec, n: int, start_idx: int
) -> list[Persona]:
    emit(
        _NODE,
        "spawn",
        f"persona-generator → segment '{seg.name}', personas "
        f"{start_idx}–{start_idx + n - 1} ({get_settings().panel_model})",
    )
    batch = structured_call(
        panel_llm(),
        PersonaBatch,
        "Generate realistic, diverse consumer personas for market research. Vary "
        "age, income, occupation, attitudes and current behaviors within the "
        "segment. Each persona must feel like a distinct real individual, not a "
        "restatement of the segment label.\n\n"
        f"Category: {stimulus.category}\n"
        f"Segment: {seg.name} - {seg.description}\n\n"
        f"Generate exactly {n} personas. Set every persona's `segment` to "
        f'"{seg.name}" and number the persona_ids "{seg.name[:4].lower()}-'
        f'{start_idx:03d}" onward.',
    )
    out = []
    for offset, p in enumerate(batch.personas[:n]):
        p.segment = seg.name
        p.persona_id = f"{seg.name[:4].lower()}-{start_idx + offset:03d}"
        out.append(p)
    return out


def _generate_segment(stimulus: Stimulus, seg: SegmentSpec) -> list[Persona]:
    chunk = get_settings().batch_size
    personas: list[Persona] = []
    while len(personas) < seg.size:
        n = min(chunk, seg.size - len(personas))
        personas.extend(_generate_chunk(stimulus, seg, n, start_idx=len(personas) + 1))
    return personas


def generate_personas(state: StudyState) -> dict:
    stimulus = state["stimulus"]
    segments = state["segments"]
    total = sum(seg.size for seg in segments)

    emit(_NODE, "start", f"Need a panel of {total} personas across {len(segments)} segments")

    key = store.cache_key(stimulus.category, segments)
    cached = store.load(key)
    if cached:
        emit(
            _NODE,
            "cache",
            f"CACHE HIT (key {key}) — loaded {len(cached)} personas from disk, 0 LLM calls",
            personas=[p.model_dump() for p in cached],
        )
        return {"personas": cached, "personas_cache_key": key}

    emit(_NODE, "cache", f"CACHE MISS (key {key}) — generating a fresh panel")
    personas: list[Persona] = []
    for seg in segments:
        personas.extend(_generate_segment(stimulus, seg))

    store.save(key, personas)
    emit(
        _NODE,
        "result",
        f"{len(personas)} personas created and cached: "
        + ", ".join(f"{p.name} ({p.age}, {p.occupation})" for p in personas[:6])
        + ("…" if len(personas) > 6 else ""),
        personas=[p.model_dump() for p in personas],
    )
    return {"personas": personas, "personas_cache_key": key}
