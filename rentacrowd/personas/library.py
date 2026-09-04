"""The persona library - a browsable folder of synthetic people.

Every persona lives as its own JSON file under `persona_library/` in the repo so
you can read, edit, diff and version them by hand. Studies recruit FROM this
library; new people are only manufactured when the library cannot cover the
requested audience (or the user explicitly asks for fresh ones).
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from rentacrowd.config import REPO_ROOT
from rentacrowd.schemas import Persona

LIBRARY_DIR = REPO_ROOT / "persona_library"
_WRITE_LOCK = threading.Lock()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "persona"


def path_for(persona: Persona) -> Path:
    return LIBRARY_DIR / f"{persona.persona_id}.json"


def ensure_dir() -> Path:
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    return LIBRARY_DIR


def load_all() -> list[Persona]:
    ensure_dir()
    out: list[Persona] = []
    for p in sorted(LIBRARY_DIR.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            out.append(Persona.model_validate_json(p.read_text()))
        except Exception:
            continue  # a hand-edited file that no longer parses shouldn't break a study
    return out


def load_by_ids(ids: list[str]) -> list[Persona]:
    wanted = set(ids)
    return [p for p in load_all() if p.persona_id in wanted]


def add(personas: list[Persona]) -> list[Persona]:
    """Write personas to the library, giving each a unique, readable id."""
    ensure_dir()
    with _WRITE_LOCK:
        existing = {p.stem for p in LIBRARY_DIR.glob("*.json")}
        saved: list[Persona] = []
        for person in personas:
            base = f"{_slug(person.segment)}--{_slug(person.name)}"
            pid, n = base, 2
            while pid in existing:
                pid, n = f"{base}-{n}", n + 1
            existing.add(pid)
            person.persona_id = pid
            path_for(person).write_text(json.dumps(person.model_dump(), indent=2))
            saved.append(person)
        write_index()
    return saved


def index() -> list[dict]:
    """A compact view of the library, small enough to put in a prompt."""
    return [
        {
            "persona_id": p.persona_id,
            "segment": p.segment,
            "who": f"{p.name}, {p.age}, {p.occupation}, {p.location}",
            "household": p.household,
            "income": p.income_band,
            "price_sensitivity": p.price_sensitivity,
            "tech_comfort": p.tech_comfort,
            "tags": p.tags,
        }
        for p in load_all()
    ]


def write_index() -> None:
    ensure_dir()
    idx = index()
    (LIBRARY_DIR / "_index.json").write_text(json.dumps(idx, indent=2))
    lines = [
        f"# Persona library ({len(idx)} people)",
        "",
        "Each person is a JSON file in this folder. Edit them by hand if you like -",
        "studies recruit from here before generating anyone new.",
        "",
        "| id | who | segment | tags |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| `{r['persona_id']}` | {r['who']} | {r['segment']} | {', '.join(r['tags'])} |"
        for r in idx
    ]
    (LIBRARY_DIR / "README.md").write_text("\n".join(lines) + "\n")


def size() -> int:
    ensure_dir()
    return len([p for p in LIBRARY_DIR.glob("*.json") if not p.name.startswith("_")])
