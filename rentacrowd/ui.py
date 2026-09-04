"""RentACrowd presentation UI.

    uv run streamlit run rentacrowd/ui.py

Give it a product, watch the LangGraph pipeline run node by node - every
sub-agent spawned, every cache hit, every LLM call is streamed live - then read
the decision-ready report.
"""

from __future__ import annotations

import os
import time

import pandas as pd
import streamlit as st

from rentacrowd.config import get_settings

# --------------------------------------------------------------------------- #
# Static pipeline description
# --------------------------------------------------------------------------- #

NODES: list[tuple[str, str, str]] = [
    ("intake", "1 · Intake", "Structure the brief into a test stimulus + market segments"),
    ("generate_personas", "2 · Persona Panel", "Build a panel of AI personas (or load it from cache)"),
    ("respond_batch", "3 · Synthetic Panel", "Parallel workers simulate each persona's reaction"),
    ("moderator", "4 · Moderator", "Qualitative debrief + probes for a real validation panel"),
    ("analyze", "5 · Analysis", "Deterministic stats + a decision-ready readout"),
]
NODE_ORDER = [n[0] for n in NODES]
NODE_META = {n[0]: (n[1], n[2]) for n in NODES}

KIND_ICON = {
    "start": "🟢",
    "cache": "💾",
    "spawn": "🌱",
    "llm": "🤖",
    "compute": "🧮",
    "result": "✅",
}

st.set_page_config(page_title="RentACrowd", page_icon="🧪", layout="wide")


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #

st.title("🧪 RentACrowd")
st.caption(
    "Synthetic market-research panels on an explicit LangGraph, served by NVIDIA NIM. "
    "A complement to real research — run many iterations fast, then validate the promising ones."
)

s = get_settings()

with st.sidebar:
    st.header("The product under test")
    name = st.text_input("Name", "MealMind")
    category = st.text_input("Category", "AI meal-planning subscription")
    price = st.text_input("Price", "$49 / month")
    how_it_works = st.text_area(
        "How it works — full description",
        "MealMind learns your household's tastes, dietary needs and budget, plans a "
        "week of meals, then auto-orders the groceries from your preferred store for "
        "delivery. It adjusts on the fly when you skip a meal or eat out.",
        height=180,
    )
    known_segments = st.text_area(
        "Target segments (optional, one per line)",
        "",
        height=80,
        placeholder="busy dual-income families\nbudget-conscious students\nhealth-focused professionals",
    )

    st.divider()
    st.subheader("Panel size")
    personas = st.slider("Personas per segment", 2, 30, s.personas_per_segment)
    batch = st.slider("Personas per LLM call", 2, 10, s.batch_size)
    st.caption(f"Model: `{s.panel_model}`  ·  ~{s.requests_per_minute} req/min cap")

    run = st.button("▶  Run study", type="primary", use_container_width=True)


def _brief() -> str:
    lines = [
        f"Product name: {name}",
        f"Category: {category}",
        f"Price: {price}",
        f"How it works: {how_it_works}",
    ]
    segs = [x.strip() for x in known_segments.splitlines() if x.strip()]
    if segs:
        lines.append("Suggested target segments: " + "; ".join(segs))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Render helpers
# --------------------------------------------------------------------------- #


def render_pipeline(slot, status: dict[str, str], events: dict[str, list[dict]], started: float):
    with slot.container():
        cols = st.columns(len(NODES))
        for (node, label, _desc), col in zip(NODES, cols):
            st_ = status.get(node, "pending")
            icon = {"pending": "⚪️", "running": "🟡", "done": "🟢"}[st_]
            col.markdown(f"### {icon}\n**{label}**")
            n_ev = len(events.get(node, []))
            if n_ev:
                col.caption(f"{n_ev} events")

        st.caption(f"⏱  {time.time() - started:0.0f}s elapsed")

        for node, label, desc in NODES:
            evs = events.get(node, [])
            st_ = status.get(node, "pending")
            with st.expander(f"{label} — {desc}", expanded=(st_ == "running")):
                if not evs:
                    st.write("_waiting…_")
                for e in evs:
                    st.markdown(f"{KIND_ICON.get(e['kind'], '•')}  {e['message']}")


def render_report(final: dict):
    report = final["report"]
    st.header("📋 Decision readout")
    st.success(f"**{report.headline}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean purchase intent", f"{report.overall_mean_purchase_intent} / 5")
    c2.metric("Positive sentiment", f"{report.overall_pct_positive}%")
    c3.metric("Personas simulated", len(final["personas"]))

    st.subheader("By segment")
    st.dataframe(
        pd.DataFrame([sr.model_dump() for sr in report.segment_results]),
        use_container_width=True,
        hide_index=True,
    )

    a, b = st.columns(2)
    with a:
        st.subheader("Key objections")
        for o in report.key_objections:
            st.markdown(f"- {o}")
        st.subheader("Recommended changes")
        for r in report.recommended_changes:
            st.markdown(f"- {r}")
    with b:
        st.subheader("Moderator debrief")
        st.write(final.get("moderator_notes", ""))
        st.subheader("Probes for a real validation panel")
        for p in final.get("moderator_probes", []):
            st.markdown(f"- {p}")

    st.caption(f"⚠️ Confidence — {report.confidence_notes}")

    with st.expander(f"The synthetic panel ({len(final['personas'])} personas)"):
        st.dataframe(
            pd.DataFrame([p.model_dump() for p in final["personas"]]),
            use_container_width=True,
            hide_index=True,
        )
    with st.expander(f"Raw persona responses ({len(final['responses'])})"):
        st.dataframe(
            pd.DataFrame([r.model_dump() for r in final["responses"]]),
            use_container_width=True,
            hide_index=True,
        )


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

if run:
    # apply panel-size choices, then rebuild settings + graph so nodes see them
    os.environ["RAC_PERSONAS_PER_SEGMENT"] = str(personas)
    os.environ["RAC_BATCH_SIZE"] = str(batch)
    get_settings.cache_clear()
    from rentacrowd.graph.build import build_graph

    graph = build_graph()

    status: dict[str, str] = {}
    events: dict[str, list[dict]] = {n: [] for n in NODE_ORDER}
    started = time.time()
    final_state: dict | None = None

    pipeline_slot = st.empty()
    render_pipeline(pipeline_slot, status, events, started)

    for mode, chunk in graph.stream(
        {"raw_product": _brief()},
        config={"max_concurrency": get_settings().max_concurrency, "recursion_limit": 100},
        stream_mode=["updates", "custom", "values"],
    ):
        if mode == "custom":
            node = chunk.get("node", "?")
            events.setdefault(node, []).append(chunk)
            if status.get(node) != "done":
                status[node] = "running"
        elif mode == "updates":
            for node in chunk:
                if node in NODE_ORDER:
                    status[node] = "done"
                    for prev in NODE_ORDER[: NODE_ORDER.index(node)]:
                        status.setdefault(prev, "done")
        elif mode == "values":
            final_state = chunk
        render_pipeline(pipeline_slot, status, events, started)

    for n in NODE_ORDER:
        status[n] = "done"
    render_pipeline(pipeline_slot, status, events, started)

    if final_state and final_state.get("report") is not None:
        render_report(final_state)
    else:
        st.error("The run finished without producing a report — check the terminal log.")
