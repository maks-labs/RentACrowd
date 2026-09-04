"""Graph state.

`StudyState` is the supervisor's shared blackboard - every worker reads from it
and writes its slice back. `PanelWorkerState` is the private state handed to each
parallel panel worker via the Send API.
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

# Workers the supervisor can dispatch to, in their natural order.
WORKERS = ["intake", "research", "recruit_panel", "run_panel", "moderator", "analyze"]


class StudyState(TypedDict, total=False):
    # --- request ---
    raw_product: str        # the product brief, free text
    request_notes: str      # e.g. "use fresh personas", "focus on rural India"
    panel_size: int         # personas wanted per segment
    competitors: list[str]  # named competitors -> the research worker scrapes these
    company: dict           # name / industry / what_they_do, for context

    # --- intake ---
    stimulus: Stimulus
    segments: list[SegmentSpec]

    # --- research ---
    market_evidence: list[dict]   # distilled real customer language per competitor

    # --- recruitment ---
    personas: list[Persona]
    recruitment_note: str   # how the panel was assembled: reused vs newly created

    # --- simulation (parallel workers append here) ---
    responses: Annotated[list[PersonaResponse], operator.add]

    # --- moderation ---
    moderator_notes: str
    moderator_probes: list[str]

    # --- output ---
    report: StudyReport
    study_dir: str          # where this study was written on disk

    # --- supervisor bookkeeping ---
    completed: Annotated[list[str], operator.add]
    supervisor_log: Annotated[list[str], operator.add]
    next_action: str


class PanelWorkerState(TypedDict):
    """One parallel panel worker's slice of the job."""

    stimulus: Stimulus
    persona_batch: list[Persona]
    evidence: str          # rendered market-evidence block (may be empty)
