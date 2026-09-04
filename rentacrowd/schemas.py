"""Pydantic models shared across the graph.

These double as the structured-output schemas for the LLM calls, so field
descriptions matter - they are part of the prompt.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Level = Literal["low", "medium", "high"]

# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


class Stimulus(BaseModel):
    """The thing being tested on the market."""

    name: str = Field(description="Product / offer name")
    category: str = Field(description="Market category, e.g. 'meal-kit subscription'")
    description: str = Field(description="What it is and what it does, 1-3 sentences")
    how_it_works: str = Field(description="The actual mechanics the user experiences")
    price: str = Field(description="Price and billing model, e.g. '$49/month'")
    key_features: list[str] = Field(default_factory=list)


class SegmentSpec(BaseModel):
    """A market segment the panel must cover."""

    name: str
    description: str = Field(description="Who is in this segment and what defines them")
    size: int = Field(description="How many personas this segment needs on the panel")


# --------------------------------------------------------------------------- #
# Personas
# --------------------------------------------------------------------------- #


class Persona(BaseModel):
    """One synthetic person.

    Deliberately detailed: thin personas produce generic, interchangeable
    answers. The concrete fields (current_solutions, frustration, voice) are what
    make a simulated reaction read like a real one.
    """

    persona_id: str
    segment: str = Field(description="Which market segment this person belongs to")
    name: str
    age: int
    gender: str = Field(description="Self-described gender")
    occupation: str = Field(description="Specific job title, not a category")
    location: str = Field(description="City/area and country, e.g. 'Leeds, UK'")
    household: str = Field(description="Who they live with, e.g. 'partner + two kids (4, 9)'")
    income_band: str = Field(description="Annual household income with currency")
    price_sensitivity: Level
    tech_comfort: Level
    psychographics: str = Field(description="Values, attitudes and lifestyle in 1-2 sentences")
    current_solutions: str = Field(
        description="What they actually use today for this need - name real products/habits"
    )
    frustration: str = Field(description="A concrete, recent, specific frustration they have")
    decision_style: str = Field(
        description="How they decide to buy, e.g. 'reads reviews for weeks, then asks partner'"
    )
    voice: str = Field(
        description="How this person talks: register, length, bluntness, slang, jargon"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="4-8 lowercase keywords for retrieval, e.g. ['parent','budget','android']",
    )


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
    switching_cost: str = Field(
        description="What they'd have to give up or change, given what they use today"
    )
    verbatim: str = Field(
        description="One or two sentences in this persona's own voice, matching their `voice` field"
    )


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
