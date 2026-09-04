"""Assemble the RentACrowd study graph as a LangGraph supervisor.

    START ─▶ supervisor ─▶ (intake | recruit_panel | run_panel | moderator | analyze)
                 ▲                                   │
                 └───────────── each worker reports back ───────────┘
    supervisor ─▶ FINISH ─▶ END

`run_panel` fans out: the edge after it emits one `Send` per persona batch to
`panel_worker`, which run in parallel and then rejoin at the supervisor.

Every worker is a named node, so the whole team is visible in LangGraph Studio.
"""

from __future__ import annotations

from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from rentacrowd.config import get_settings
from rentacrowd.events import emit
from rentacrowd.graph.nodes.analyze import analyze
from rentacrowd.graph.nodes.intake import intake
from rentacrowd.graph.nodes.moderator import moderator
from rentacrowd.graph.nodes.recruit_panel import recruit_panel
from rentacrowd.graph.nodes.research import evidence_block, research
from rentacrowd.graph.nodes.run_panel import panel_worker, run_panel
from rentacrowd.graph.nodes.supervisor import supervisor
from rentacrowd.state import StudyState

_WORKERS = ["intake", "research", "recruit_panel", "run_panel", "moderator", "analyze"]


def _route(state: StudyState):
    """Supervisor's decision -> the next node (or the parallel fan-out, or END)."""
    action = state.get("next_action", "FINISH")

    if action == "run_panel":
        return "run_panel"
    if action in _WORKERS:
        return action
    return END


def _fan_out(state: StudyState) -> list[Send]:
    bs = get_settings().batch_size
    personas = state.get("personas") or []
    stimulus = state.get("stimulus")
    if not personas or stimulus is None:
        return ["supervisor"]
    evidence = evidence_block(state.get("market_evidence") or [])
    sends = [
        Send("panel_worker", {
            "stimulus": stimulus,
            "persona_batch": personas[i : i + bs],
            "evidence": evidence,
        })
        for i in range(0, len(personas), bs)
    ]
    emit(
        "run_panel",
        "spawn",
        f"fanning {len(personas)} personas into {len(sends)} parallel panel workers",
    )
    return sends


def build_graph():
    g = StateGraph(StudyState)

    g.add_node("supervisor", supervisor)
    g.add_node("intake", intake)
    g.add_node("research", research)
    g.add_node("recruit_panel", recruit_panel)
    g.add_node("run_panel", run_panel)
    g.add_node("panel_worker", panel_worker)
    g.add_node("moderator", moderator)
    g.add_node("analyze", analyze)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor", _route, [*_WORKERS, END]
    )

    # every worker reports back to the supervisor…
    for w in ["intake", "research", "recruit_panel", "moderator", "analyze"]:
        g.add_edge(w, "supervisor")
    # …except run_panel, which fans out first, then the workers rejoin.
    g.add_conditional_edges("run_panel", _fan_out, ["panel_worker", "supervisor"])
    g.add_edge("panel_worker", "supervisor")

    return g.compile()


graph = build_graph()
