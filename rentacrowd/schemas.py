"""Pydantic models shared across the graph.

These double as the `with_structured_output` schemas for the LLM calls, so the
field descriptions matter - they are part of the prompt.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


class Stimulus(BaseModel):
    """The thing being tested on the market."""

    name: str = Field(description="Product / offer name")
    category: str = Field(description="Market category, e.g. 'meal-kit subscription'")
    description: str = Field(description="What it is and what it does, 1-3 sentences")
    price: str = Field(description="Price and billing model, e.g. '$49/month'")
    key_features: list[str] = Field(default_factory=list)
    target_segments: list[str] = Field(
        default_factory=list,
        description="Named market segments to simulate, e.g. 'budget-conscious students'",
    )


class SegmentSpec(BaseModel):
    """A market segment we will populate with personas."""

    name: str
    description: str = Field(description="Who is in this segment and what defines them")
    size: int = Field(description="How many personas to generate for this segment")


# --------------------------------------------------------------------------- #
# Personas
# --------------------------------------------------------------------------- #


class Persona(BaseModel):
    persona_id: str
    segment: str
    name: str
    age: int
    occupation: str
    location: str
    income_band: str
    psychographics: str = Field(description="Values, attitudes, lifestyle in 1-2 sentences")
    relevant_behaviors: str = Field(
        description="Category-relevant habits, current solutions, past purchases"
    )
    price_sensitivity: Literal["low", "medium", "high"]


class PersonaBatch(BaseModel):
    """Structured-output wrapper: the generator returns many personas per call."""

    personas: list[Persona]


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #


class PersonaResponse(BaseModel):
    persona_id: str
    purchase_intent: int = Field(ge=0, le=5, description="0=no interest at all, 5=definitely yes")
    willingness_to_pay: str = Field(description="Max price this persona would pay, with unit")
    sentiment: Literal["negative", "mixed", "positive"]
    top_objection: str = Field(description="The single biggest reason they hesitate")
    what_would_convince: str = Field(description="What would move them to buy")
    verbatim: str = Field(description="A one-sentence quote in the persona's own voice")


class PanelResponseBatch(BaseModel):
    responses: list[PersonaResponse]


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #


class SegmentResult(BaseModel):
    segment: str
    n: int
    mean_purchase_intent: float
    pct_positive: float = Field(description="Share of personas with positive sentiment, 0-100")
    top_objections: list[str]
    wtp_summary: str


class StudyReport(BaseModel):
    headline: str = Field(description="One-line verdict for a decision-maker")
    overall_mean_purchase_intent: float
    overall_pct_positive: float
    segment_results: list[SegmentResult]
    key_objections: list[str] = Field(description="Ranked across the whole panel")
    recommended_changes: list[str]
    confidence_notes: str = Field(
        description="Caveats: where the synthetic panel is least trustworthy for this study"
    )
