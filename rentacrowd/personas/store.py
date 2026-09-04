"""Disk cache for generated persona panels.

Persona generation is a one-time cost per (category, segment-set). Studies that
reuse the same panel pay zero LLM calls for this layer.
"""

from __future__ import annotations

import hashlib
import json

from rentacrowd.config import get_settings
from rentacrowd.schemas import Persona, SegmentSpec


def cache_key(category: str, segments: list[SegmentSpec]) -> str:
    payload = json.dumps(
        {"category": category.lower().strip(),
         "segments": sorted((s.name, s.description, s.size) for s in segments)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load(key: str) -> list[Persona] | None:
    path = get_settings().personas_dir / f"{key}.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return [Persona.model_validate(p) for p in raw]


def save(key: str, personas: list[Persona]) -> None:
    path = get_settings().personas_dir / f"{key}.json"
    path.write_text(json.dumps([p.model_dump() for p in personas], indent=2))
