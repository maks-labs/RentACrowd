"""Graph state.

`StudyState` is the top-level channel set. `BatchState` is the per-worker state
handed to each parallel `respond_batch` node via the Send API.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from rentacrowd.schemas import (
    Persona,
    PersonaResponse,
    SegmentSpec,
    Stimulus,
    StudyReport,
)


class StudyState(TypedDict, total=False):
    # inputs
    raw_product: str            # free-text product description from the user
    stimulus: Stimulus          # structured by the intake node
    segments: list[SegmentSpec]

    # persona layer
    personas_cache_key: str
    personas: list[Persona]

    # simulation layer - responses accumulate from parallel workers
    responses: Annotated[list[PersonaResponse], operator.add]

    # moderator layer
    moderator_probes: list[str]
    moderator_notes: str

    # output
    report: StudyReport


class BatchState(TypedDict):
    """One parallel persona-panel worker's slice of the job."""

    stimulus: Stimulus
    persona_batch: list[Persona]
