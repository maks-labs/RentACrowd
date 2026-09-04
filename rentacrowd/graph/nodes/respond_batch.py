"""One parallel persona-panel worker.

Handed a slice of personas via the Send API. Makes a SINGLE structured LLM call
that role-plays every persona in the slice against the stimulus, then appends the
responses to the shared `responses` channel (reducer = list concat).
"""

from __future__ import annotations

from rentacrowd.config import get_settings
from rentacrowd.events import emit
from rentacrowd.llm import panel_llm, structured_call
from rentacrowd.schemas import PanelResponseBatch
from rentacrowd.state import BatchState

_NODE = "respond_batch"


def _invoke(persona_block: str, stimulus_block: str) -> PanelResponseBatch:
    return structured_call(
        panel_llm(),
        PanelResponseBatch,
        "Simulate a consumer panel. Respond AS EACH persona independently - stay in "
        "their situation, budget and attitudes. Do not make personas agree with each "
        "other. Be willing to give low purchase intent and hard objections; most new "
        "products are met with indifference.\n\n"
        f"STIMULUS\n{stimulus_block}\n\n"
        f"PERSONAS\n{persona_block}\n\n"
        "Return one response object per persona, matching persona_id.",
    )


def respond_batch(state: BatchState) -> dict:
    stimulus = state["stimulus"]
    personas = state["persona_batch"]

    stimulus_block = (
        f"{stimulus.name} ({stimulus.category}) - {stimulus.description}\n"
        f"Price: {stimulus.price}\n"
        f"Features: {', '.join(stimulus.key_features) or 'n/a'}"
    )
    persona_block = "\n".join(
        f"[{p.persona_id}] {p.name}, {p.age}, {p.occupation}, {p.location}, "
        f"income {p.income_band}, price-sensitivity {p.price_sensitivity}. "
        f"{p.psychographics} {p.relevant_behaviors}"
        for p in personas
    )

    ids = ", ".join(p.persona_id for p in personas)
    emit(
        _NODE,
        "spawn",
        f"panel worker simulating {len(personas)} personas [{ids}] "
        f"against '{stimulus.name}' ({get_settings().panel_model})",
    )
    out = _invoke(persona_block, stimulus_block)
    emit(
        _NODE,
        "result",
        " · ".join(
            f"{r.persona_id}: intent {r.purchase_intent}/5 ({r.sentiment})"
            for r in out.responses
        ),
        responses=[r.model_dump() for r in out.responses],
    )
    return {"responses": out.responses}
