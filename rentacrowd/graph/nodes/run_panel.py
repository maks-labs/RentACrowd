"""Run the panel.

`run_panel` is the dispatcher the supervisor calls; it hands the shared state
straight back (the fan-out happens on the edge, see graph/build.py). `panel_worker`
is one parallel worker: it role-plays a slice of the panel against the stimulus,
carrying each persona's full detail into the prompt so answers stay in-character.
"""

from __future__ import annotations

from rentacrowd.config import get_settings
from rentacrowd.events import emit
from rentacrowd.llm import panel_llm, structured_call
from rentacrowd.schemas import PanelResponseBatch
from rentacrowd.state import PanelWorkerState, StudyState

_NODE = "run_panel"


def run_panel(state: StudyState) -> dict:
    emit(
        _NODE,
        "start",
        f"Dispatching {len(state['personas'])} personas to parallel panel workers",
    )
    return {"completed": [_NODE]}


def _persona_card(p) -> str:
    return (
        f"[{p.persona_id}] {p.name} — {p.age}, {p.gender}, {p.occupation}, {p.location}\n"
        f"  household: {p.household} · income: {p.income_band} · "
        f"price-sensitivity: {p.price_sensitivity} · tech: {p.tech_comfort}\n"
        f"  values: {p.psychographics}\n"
        f"  uses today: {p.current_solutions}\n"
        f"  recent frustration: {p.frustration}\n"
        f"  decides by: {p.decision_style}\n"
        f"  voice: {p.voice}"
    )


def panel_worker(state: PanelWorkerState) -> dict:
    stimulus = state["stimulus"]
    personas = state["persona_batch"]
    evidence = state.get("evidence", "")
    ids = ", ".join(p.persona_id for p in personas)

    emit(
        "run_panel",
        "spawn",
        f"panel worker · {len(personas)} personas [{ids}] vs "
        f"'{stimulus.name}' ({get_settings().panel_model})",
    )

    stim = (
        f"{stimulus.name} ({stimulus.category})\n"
        f"{stimulus.description}\n"
        f"How it works: {stimulus.how_it_works}\n"
        f"Price: {stimulus.price}\n"
        f"Features: {', '.join(stimulus.key_features) or 'n/a'}"
    )
    cards = "\n\n".join(_persona_card(p) for p in personas)

    out = structured_call(
        panel_llm(),
        PanelResponseBatch,
        "Simulate a consumer panel. Answer AS EACH persona, staying inside their "
        "situation, budget, habits and VOICE. Reference what they use today and "
        "their frustration where relevant. Personas must not converge - keep their "
        "disagreements. Low purchase intent and blunt objections are expected; most "
        "new products are met with indifference.\n\n"
        f"STIMULUS\n{stim}\n\n"
        + (f"{evidence}\n\n" if evidence else "")
        + f"PANEL\n{cards}\n\n"
        "Return exactly one response per persona, matched by persona_id.",
    )

    emit(
        "run_panel",
        "result",
        " · ".join(
            f"{r.persona_id}: {r.purchase_intent}/5 ({r.sentiment})" for r in out.responses
        ),
        responses=[r.model_dump() for r in out.responses],
    )
    return {"responses": out.responses}
