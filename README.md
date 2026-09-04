<img width="1800" height="1012" alt="Screenshot 2026-09-04 at 11 45 33 PM" src="https://github.com/user-attachments/assets/40641d4d-a08f-4fca-a735-b4c68dd943ae" />

<div align="center">

# RentACrowd

**A synthetic market-research panel, run by a multi-agent supervisor — grounded in real customer language, not model priors.**

Describe a product. Six agents plan the study, cast a panel of persistent synthetic people,
argue about it in parallel, and hand you a decision — in about the time it takes to make coffee.

[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-1C3C3C)](https://github.com/langchain-ai/langgraph)
[![LangChain Core](https://img.shields.io/badge/agent%20runtime-LangChain%20Core-1C3C3C)](https://github.com/langchain-ai/langchain)
[![NVIDIA NIM](https://img.shields.io/badge/inference-NVIDIA%20NIM-76B900)](https://build.nvidia.com/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/serving-FastAPI%20%2B%20SSE-009688)](https://fastapi.tiangolo.com/)
[![LangGraph Studio](https://img.shields.io/badge/debugging-LangGraph%20Studio-1C3C3C)](https://github.com/langchain-ai/langgraph-studio)
[![GSAP](https://img.shields.io/badge/scroll%20engine-GSAP%20%2B%20ScrollTrigger-88CE02)](https://gsap.com/)
[![Motion One](https://img.shields.io/badge/UI%20animation-Motion%20One-black)](https://motion.dev/)
[![Server-Sent Events](https://img.shields.io/badge/live%20updates-SSE-8A2BE2)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)

</div>

---

## The problem this exists to solve

Real customer research is **slow** (6–8 weeks, screen to readout), **expensive** ($15k+ per
professional round), and **small-sample by necessity** — a moderator can only run so many
sessions. The consequence isn't that teams skip research; it's that they get *one shot* at it,
long after the decision that actually mattered was made, and never re-test as the product or the
market shifts.

RentACrowd doesn't claim to replace that. It exists to compress the *first ninety percent* of the
loop — narrowing ten rough ideas down to the two worth taking into the field — so the real budget
and the real humans go to validating a shortlist instead of exploring a wide-open space blind.

> Synthetic research is a **complement, not a replacement** for fieldwork. Run many iterations
> fast and cheap; validate the promising ones with real people, using the moderator's own probe
> questions as your interview guide.

---

## The team — an explicit LangGraph supervisor

Every "agent" here is a real, named node in a `StateGraph` — not a tool call hidden inside a
black-box framework. You can open [LangGraph Studio](#inspect-the-graph) and watch the supervisor
route between them live, which was the whole point of building this on raw LangGraph instead of a
higher-level agent framework.

```
                              ┌───────────────┐
                    ┌────────▶│   supervisor   │◀────────┐
                    │         └───────┬───────┘          │
        every worker reports back    │ decides the next   node to run,
        (and the loop continues)     │ reacting to what came back so far
                    │                ▼                    │
      ┌─────────┬─────────┬──────────────┬───────────┬─────────┐
      │         │         │              │           │         │
   intake   research   recruit_panel  run_panel   moderator  analyze
      │         │             │           │
      │         │             │           ▼
      │         │             │     ┌─────────────────────────┐
      │         │             │     │   Send() fan-out (n)    │
      │         │             │     │  panel_worker × batches │  ← runs in parallel
      │         │             │     └─────────────────────────┘
      ▼         ▼             ▼
  structures  scrapes      casts the
  the brief   evidence      panel
```

| Node | What it actually does | Reports to the supervisor |
|---|---|---|
| **`supervisor`** | An LLM router with a deterministic guard rail underneath it (`_forced_move`) — it decides which worker runs next based on what's already in state, not a fixed pipeline order. | — |
| **`intake`** | Turns your free-text brief into a structured `Stimulus` (price, features, positioning) and exactly N `SegmentSpec`s worth testing separately. | `stimulus`, `segments` |
| **`research`** | Scrapes **real** App Store reviews, Hacker News threads, and (optionally) Reddit for the competitors you name, and distils the recurring objections and phrasing into an evidence pack. Runs only when you name competitors; degrades gracefully to nothing found. | `market_evidence` |
| **`recruit_panel`** | Casts the panel — see [Personas](#personas--how-they-are-generated-and-cast) below. | `personas`, `recruitment_note` |
| **`run_panel`** | Dispatches the cast panel; the graph edge fans it out into parallel `panel_worker` calls via LangGraph's `Send` API, one call per batch of personas, then rejoins at the supervisor. | `responses` (accumulated) |
| **`moderator`** | Reads every reaction and drafts open, probing follow-ups like a real focus-group lead — surfaces disagreement and contradictions instead of averaging them away. | `moderator_notes`, `moderator_probes` |
| **`analyze`** | Turns every reaction and note into one decision-ready `StudyReport`: scored intent, ranked objections, a plain-language call. | `report` |

The supervisor's own state (`StudyState` in `rentacrowd/state.py`) is a shared blackboard — every
worker reads the slice it needs and writes its own slice back with `Annotated[..., operator.add]`
for the fields that accumulate (like `responses`, which is written by N parallel workers at once).

---

## Personas — how they are generated, and how they work

This is the part that decides whether the whole exercise is worth anything. A **thin** persona —
"Sarah, 34, marketing manager, likes convenience" — produces a **generic** reaction, because
there's nothing in the prompt to pull the model away from its own average. RentACrowd fights that
in three ways: detailed schemas, an explicit anti-genericism prompt, and grounding in scraped
real-world language.

### 1. The schema forces specificity

Every persona (`Persona` in `rentacrowd/schemas.py`) carries 17 fields, and the ones that matter
most are the concrete, narrative ones — not the demographic ones:

```python
current_solutions   # names REAL products/habits they use today, not a category
frustration          # ONE recent, specific incident — a number, a time, a place
decision_style       # how they actually decide to buy, e.g. "reads reviews for
                      #   weeks, then asks partner"
voice                 # register, length, bluntness, slang — how they'd actually type
tags                  # 4-8 keywords used later for retrieval (see casting, below)
```

Field *descriptions* in the schema aren't just documentation — they're injected directly into the
structured-output prompt (`structured_call` in `rentacrowd/llm.py`), so a vague description
produces a vague persona field. That's why they read like engineering constraints, not comments.

### 2. Generation actively fights the model's pull toward archetypes

`rentacrowd/personas/generate.py` wraps every generation call in an explicit rule set
(`_ANTI_GENERIC`) that the LLM must follow: real ordinary names (never a famous person or an
alliterative joke name), named real products/habits per country and income, one *concrete* recent
frustration with a real detail, spread ages/incomes/household shapes across a batch instead of
clustering, and voices that genuinely differ person to person. Personas are generated in small
batches (`RAC_BATCH_SIZE`, default 4) rather than one giant call — smaller batches keep the model
from converging on a single "type" partway through the response.

### 3. Reactions are grounded in scraped reality, not vibes

When you name competitors, the `research` worker pulls **real** customer language before the panel
ever reacts — iTunes/App Store ratings and review text, Hacker News discussion via the Algolia
search API, and Reddit if you've configured OAuth credentials (all free, all keyless except
Reddit). It distils the recurring objections and exact phrasing into an evidence block that gets
threaded into every panel worker's prompt. So when a synthetic persona objects to your pricing, the
objection can trace back to an actual sentence a real person wrote about a real competitor —
that's the difference between *"probably some people would find this expensive"* and a panel that
argues with the market's actual, specific complaints.

### How a reaction actually gets produced

`panel_worker` (`rentacrowd/graph/nodes/run_panel.py`) is the unit of simulation. Each call:

1. Receives a **batch** of personas (not the whole panel — see [fan-out](#the-team--an-explicit-langgraph-supervisor) above) plus the stimulus and the evidence block.
2. Builds a full "persona card" per person — demographics, values, what they use today, their
   frustration, how they decide, their voice — so the model has no excuse to flatten them into one
   voice.
3. Is explicitly instructed **not** to let personas converge: disagreement and blunt objections are
   expected, because most new products are, correctly, met with indifference by most people.
4. Returns one structured `PersonaResponse` per persona — purchase intent (0–5), willingness to
   pay, sentiment, the single biggest objection, what would convince them, and their switching
   cost given what they use today.

Because every batch runs as a separate parallel `Send`, a 24-person panel in 6 batches of 4 is six
concurrent LLM calls, not twenty-four sequential ones — that's the actual reason a full study
finishes in roughly a minute instead of twenty.

---

## Storage — everything is a plain file, on purpose

There is no database here. That's deliberate: the whole point of an internal research tool is that
someone can `cat`, `diff`, hand-edit, or `git blame` the exact input that produced a given result.

```
RentACrowd/
├── persona_library/              ← the reusable population (tracked in git)
│   ├── <segment>--<name>.json    ← one file per synthetic person, human-editable
│   ├── _index.json               ←   compact machine index, regenerated on every write
│   └── README.md                 ←   auto-generated table, browsable on GitHub
│
└── studies/                      ← one folder per run (gitignored — these are outputs, not source)
    └── <timestamp>--<slug>/
        ├── stimulus.json         ← the structured brief this run tested
        ├── panel.json            ← exactly which personas were recruited, and from where
        ├── responses.json        ← every raw persona reaction
        ├── report.json           ← the final structured StudyReport
        └── report.md             ← the same report, human-readable
```

**The persona library is the interesting part.** `recruit_panel` doesn't generate a fresh panel
every time you run a study — it **casts from the library first**:

1. `_cast_from_library` scores every existing persona against every requested segment by weighted
   keyword overlap between the segment's description and the persona's own words (occupation,
   household, psychographics, frustration, tags — tags weighted 3× because they're purpose-built
   for retrieval). This is deterministic and has **no LLM call**, so casting a 100-person library
   against 4 segments takes milliseconds, not a model round-trip.
2. Only the **shortfall** — segments the library genuinely can't cover — gets handed to
   `make_personas` to manufacture new people.
3. Newly created people are **written straight back** to `persona_library/`, so the population
   compounds. Run ten studies and you don't have ten disposable panels; you have one population
   that's been asked ten different questions and keeps getting more useful to reuse.

This matters for a reason beyond speed: **consistency across studies**. If the same "budget-
conscious parent of two" persona reacts to your pricing change in March and your feature launch in
June, that's a comparable signal over time — a freshly generated persona every run would just be
noise dressed up as data.

```bash
uv run rentacrowd-seed --per 4      # seed ~100 people across 25 audience slices, ~4 min, one-time
```

---

## Where this gets meaningfully better: CDP integration

Right now every persona is grounded in **public** signal — scraped reviews and forum language
about your named competitors. That's real, but it's still a proxy for *your* actual customers. The
next lever is plugging in a **Customer Data Platform** (Segment, mParticle, RudderStack, or
similar) as a second grounding source, alongside the scraper — not replacing it.

**What breaks without it, concretely:**

- Personas are demographically plausible but **behaviorally invented** — `current_solutions` and
  `frustration` are the model's best guess at what a segment like yours does, not what your
  segment actually does.
- There's no way to weight the panel by your **real** customer mix. If 60% of your actual base is
  one behavioral cluster, a synthetic panel built from public review language has no way to know
  that and will effectively treat every segment as equally likely.
- You can't validate a synthetic finding against what already happened. If the panel says a price
  increase would tank purchase intent, there's no closed loop back to whether your real churn data
  agrees.

**What a CDP integration would concretely change, node by node:**

| Node | Today | With CDP grounding |
|---|---|---|
| `intake` | Segments are invented from the brief alone | Segments can be proposed **from your actual customer clusters** (behavioral cohorts, RFM tiers, lifecycle stage) instead of guessed from a text description |
| `recruit_panel` | Casts from a library seeded by generic audience slices | Casts (or generates) personas whose `current_solutions`, `price_sensitivity`, and `decision_style` are anchored to real event streams — actual purchase history, actual feature usage, actual support-ticket themes, pulled per-cohort and anonymized/aggregated before it ever reaches a prompt | 
| `research` | Evidence is public review/forum language about competitors | Evidence pack extends to **first-party** signal: your own product-analytics events (drop-off points, feature adoption, churn-adjacent behavior) alongside public competitor chatter |
| `analyze` | The panel's synthetic purchase intent is the only signal in the readout | The readout can cite a **real base rate** next to the synthetic one — "the panel predicts X; your closest real cohort's actual conversion on a comparable change was Y" — turning a plausibility check into a calibration check |

This is explicitly a **Phase 3 roadmap item**, not a rewrite: the engine (supervisor, fan-out,
persona schema) doesn't change. A CDP integration is a new evidence source feeding the same
`market_evidence` slot that `research` already populates, plus an optional cohort-aware casting
path in `recruit_panel` — additive, not architectural.

---

## Setup

```bash
uv sync
cp .env.example .env                # add NVIDIA_API_KEY (nvapi-...)
uv run rentacrowd-seed               # populate the library (~4 min, one-time)
```

## Run — the UI

```bash
uv run rentacrowd-ui                 # http://127.0.0.1:8000
```

The landing page tells the whole story as you scroll — the architecture, how each agent works, and
what's grounding it — before a 3-step form takes the company, the product, and its competitors. On
submit it flies into the study view, where the supervisor dispatches work between agents in a live
3D constellation: a packet travels from the supervisor to each worker and back, fan-out workers
spring in as they run, and the decision readout rises as a bottom sheet when `analyze` finishes.

## Run — terminal

```bash
uv run rentacrowd "A $49/mo AI meal planner that auto-orders groceries" \
  --notes "focus on rural households"
```

## Inspect the graph

```bash
uv run langgraph dev --no-reload
```

Opens LangGraph Studio against the live graph — every node above is a real, inspectable step, and
you can watch the supervisor's routing decisions and each worker's input/output in real time.

## Speed

Every LLM step calls the free `nemotron-3.5-lightning` model (~28 tok/s). Library casting is
deterministic (no LLM call at all), and the scrapers are fast, so a small panel against a
fully-seeded library runs in **~80 seconds**. Bigger panels add roughly one parallel panel-worker
call per batch, not one per persona. Tune panel size, batch size, and models in `.env` —
see `.env.example` for every knob.
