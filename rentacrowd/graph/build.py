"""Assemble the RentACrowd study graph.

    START
      -> intake              structure the product into a test stimulus + segments
      -> generate_personas   build (or load cached) the persona panel
      -> [fan_out]           Send API: split personas into batches
           => respond_batch  N parallel panel workers (concurrency-capped at invoke time)
      -> moderator           qualitative debrief + probes for a real validation panel
      -> analyze             deterministic stats + decision-ready readout
    END

Every stage is an explicit, named node so the whole pipeline is visible in
LangGraph Studio (`langgraph dev`).
"""

from __future__ import annotations

from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from rentacrowd.config import get_settings
from rentacrowd.events import emit
from rentacrowd.graph.nodes.analyze import analyze
from rentacrowd.graph.nodes.generate_personas import generate_personas
from rentacrowd.graph.nodes.intake import intake
from rentacrowd.graph.nodes.moderator import moderator
from rentacrowd.graph.nodes.respond_batch import respond_batch
from rentacrowd.state import StudyState


def _fan_out(state: StudyState) -> list[Send]:
    """Split the persona panel into batches, one Send per parallel worker.

    Uses `.get` so that re-evaluating the edge against a partial state (e.g. after
    a run is cancelled in Studio) returns no work instead of raising KeyError.
    """
    bs = get_settings().batch_size
    personas = state.get("personas") or []
    stimulus = state.get("stimulus")
    if not personas or stimulus is None:
        return []
    sends = [
        Send("respond_batch", {"stimulus": stimulus, "persona_batch": personas[i : i + bs]})
        for i in range(0, len(personas), bs)
    ]
    emit(
        "generate_personas",
        "spawn",
        f"fanning out {len(personas)} personas into {len(sends)} parallel panel workers",
    )
    return sends


def build_graph():
    g = StateGraph(StudyState)

    g.add_node("intake", intake)
    g.add_node("generate_personas", generate_personas)
    g.add_node("respond_batch", respond_batch)
    g.add_node("moderator", moderator)
    g.add_node("analyze", analyze)

    g.add_edge(START, "intake")
    g.add_edge("intake", "generate_personas")
    g.add_conditional_edges("generate_personas", _fan_out, ["respond_batch"])
    g.add_edge("respond_batch", "moderator")
    g.add_edge("moderator", "analyze")
    g.add_edge("analyze", END)

    return g.compile()


graph = build_graph()
