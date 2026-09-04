# RentACrowd

Synthetic market-research panels driven by a **LangGraph supervisor**, served by NVIDIA NIM.

Give it a product; a supervisor agent runs the study — structuring the brief,
scraping real customer language about your competitors, recruiting a persona
panel, fanning out parallel workers to simulate every persona's reaction,
debriefing as a moderator, and writing a decision readout.

> Synthetic research is a **complement, not a replacement** for real fieldwork.
> Run many iterations fast, then validate the promising ones with the moderator's
> probe questions.

## The team

```
              ┌─────────────┐
   START ────▶│  supervisor  │◀──── every worker reports back
              └─────────────┘
                    │ decides who runs next (reacts to your notes)
   ┌────────┬───────┼────────┬───────────┬──────────┐
 intake  research  recruit  run_panel  moderator  analyze
                    panel       │
                          panel_worker ×N  (parallel, Send API)
```

Every worker is a named node — the whole team is visible in LangGraph Studio.

- **research** — scrapes App Store reviews + Hacker News (and Reddit, if you set
  `REDDIT_CLIENT_ID`/`REDDIT_CLIENT_SECRET`) for the competitors you name,
  distils the real objections and vocabulary, and feeds them into every panel
  worker so reactions echo real customer language. Runs only when competitors
  are named; degrades gracefully when nothing is found.
- **recruit_panel** — matches library personas to each segment by keyword
  overlap (deterministic, instant) and only generates the shortfall.

## Persona library

`persona_library/` holds a reusable population — one JSON file per synthetic
person, hand-editable, versioned. Studies **recruit from here first** and only
manufacture new people for the shortfall (or when your notes ask for fresh ones).
New hires are written back, so the population compounds.

```bash
uv run rentacrowd-seed --per 4      # ~100 people across 25 audience slices (parallel, ~4 min)
```

## Setup

```bash
uv sync
cp .env.example .env                # add NVIDIA_API_KEY (nvapi-...)
uv run rentacrowd-seed              # populate the library (~4 min, one-time)
```

## Run — the UI

```bash
uv run rentacrowd-ui               # http://127.0.0.1:8000
```

A landing page explains the system; a 3-step form takes the company, the product,
and its competitors. On submit it flies into the study view, where the supervisor
dispatches work between agents in a 3D constellation — a packet travels from the
supervisor to each worker and back, the fan-out workers spring in, and the
decision readout rises as a sheet. Studies are saved to `studies/<timestamp>--<name>/`.

## Run — terminal

```bash
uv run rentacrowd "A $49/mo AI meal planner that auto-orders groceries" \
  --notes "focus on rural households"
```

## Inspect the graph

```bash
uv run langgraph dev --no-reload
```

## Speed

Every LLM step is a call to the free `nemotron-3.5-lightning` model (~28 tok/s).
Library recruitment is deterministic (no LLM), and the scrapers are fast, so a
small-panel study with a fully-seeded library runs in **~80 seconds**. Bigger
panels add one panel-worker call per batch. Tune panel size and models in `.env`.
