# RentACrowd

Synthetic market-research panels on an explicit LangGraph, served by NVIDIA NIM.

Give it a product; it structures a test stimulus, builds a panel of AI personas
across market segments, simulates their reactions in parallel, runs a moderator
debrief, and produces a decision-ready readout (purchase intent, WTP, objections,
segment cuts).

> Synthetic research is a **complement, not a replacement** for real fieldwork.
> Use it to run 10x more iterations before spending on a human panel, then
> validate the promising ones with the moderator's probe questions.

## The graph

```
START
  -> intake              free-text product -> structured Stimulus + segments
  -> generate_personas   build (or load cached) the persona panel
  -> respond_batch  x N   parallel panel workers (Send API, concurrency-capped)
  -> moderator           qualitative debrief + probes for a real validation panel
  -> analyze             deterministic stats + LLM-written readout
END
```

Every stage is a named node, visible in LangGraph Studio.

### Why not `deepagents` for the core?

The pipeline is structured, not open-ended, so an explicit graph gives us:
visible sub-agents, controllable fan-out width, predictable cost, and
reproducible runs under the NIM ~40 req/min ceiling. `deepagents` is reserved
for a Phase-2 research-powered `intake` node.

## Setup

```bash
uv sync
cp .env.example .env      # add your NVIDIA_API_KEY (nvapi-...)
```

## Run — presentation UI (recommended)

```bash
uv run streamlit run rentacrowd/ui.py
```

Enter the product and its full working, hit **Run study**, and watch the pipeline
execute node by node — every sub-agent spawned, every persona-cache hit, every
LLM call is streamed live under its node — then read the decision readout, the
synthetic panel, and the raw responses.

## Run — terminal

```bash
uv run rentacrowd "A $49/mo AI meal planner that auto-orders your groceries"
```

Writes a full JSON transcript to `~/.rentacrowd/studies/`. Persona panels are
cached in `~/.rentacrowd/personas/` and reused across studies (0 LLM calls on a hit).

## Inspect / debug the graph

```bash
uv run langgraph dev --no-reload      # opens LangGraph Studio
```

## Throughput

Sub-agents are graph nodes, not extra LLM calls. One `respond_batch` call
role-plays `RAC_BATCH_SIZE` personas at once, so a 200-persona study is ~20
calls. A shared process-wide rate limiter keeps the combined rate under
`RAC_REQUESTS_PER_MINUTE`. Tune everything in `.env`.

The free NIM nemotron models are "thinking" models; RentACrowd disables that
(`chat_template_kwargs={"thinking": false}`) so calls return in seconds instead
of tens of seconds. A full study at demo size is ~2-3 minutes.
