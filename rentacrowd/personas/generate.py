"""Manufacture new synthetic people.

Used both to seed the library and to top it up mid-study when the requested
audience isn't covered. The prompt fights the model's pull toward archetypes:
generic personas produce generic, interchangeable answers, which is the single
biggest credibility problem in synthetic research.
"""

from __future__ import annotations

from rentacrowd.config import get_settings
from rentacrowd.llm import panel_llm, structured_call
from rentacrowd.schemas import Persona, PersonaBatch

_ANTI_GENERIC = (
    "Rules, all mandatory:\n"
    "- Real, ordinary full names that fit the person's stated place and background. "
    "NEVER use names of famous people, song titles, fictional characters, or "
    "alliterative joke names.\n"
    "- Name the ACTUAL products, apps, shops or habits each person uses today. Be "
    "specific and plausible for their country and income.\n"
    "- `frustration` must be one concrete recent incident with a real detail - a "
    "number, a time of day, a place - not a general complaint.\n"
    "- Spread ages, incomes, genders and household shapes across the group. Do not "
    "cluster everyone at the same age or life stage.\n"
    "- `voice` must genuinely differ per person: some blunt and short, some rambling, "
    "some formal, some heavy on slang or jargon.\n"
    "- These are ordinary people, not marketing archetypes. Give them contradictions."
)


def make_personas(segment_name: str, brief: str, n: int) -> list[Persona]:
    """Generate `n` personas for one audience slice, in chunks the model handles well."""
    chunk = max(2, get_settings().batch_size)
    out: list[Persona] = []
    while len(out) < n:
        want = min(chunk, n - len(out))
        batch = structured_call(
            panel_llm(),
            PersonaBatch,
            f"Create {want} realistic, distinct consumer personas for market research.\n\n"
            f"Audience slice: {segment_name}\n"
            f"Definition: {brief}\n\n"
            f"{_ANTI_GENERIC}\n\n"
            f'Set every persona\'s `segment` field to exactly "{segment_name}".',
        )
        for person in batch.personas[:want]:
            person.segment = segment_name
            out.append(person)
    return out
