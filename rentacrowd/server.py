"""FastAPI backend for the RentACrowd web UI.

    uv run rentacrowd-ui           # http://127.0.0.1:8000

Serves the single-page app and streams the supervisor graph's events over
Server-Sent Events so the frontend can animate work moving between agents.
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from rentacrowd.config import get_settings
from rentacrowd.graph.build import build_graph
from rentacrowd.personas import library

WEB = Path(__file__).parent / "web"

app = FastAPI(title="RentACrowd")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/library")
def library_stats() -> dict:
    idx = library.index()
    segs: dict[str, int] = {}
    for r in idx:
        segs[r["segment"]] = segs.get(r["segment"], 0) + 1
    return {"count": len(idx), "segments": segs}


class StudyRequest(BaseModel):
    # company
    company_name: str = ""
    industry: str = ""
    company_does: str = ""
    # product
    name: str = ""
    category: str = ""
    price: str = ""
    how_it_works: str = ""
    # market context
    competitors: str = ""      # comma-separated
    notes: str = ""
    panel_size: int = 0


def _brief(r: StudyRequest) -> str:
    return "\n".join(
        p for p in [
            f"Company: {r.company_name}" if r.company_name else "",
            f"Industry: {r.industry}" if r.industry else "",
            f"What the company does: {r.company_does}" if r.company_does else "",
            "---",
            f"Product name: {r.name}" if r.name else "",
            f"Category: {r.category}" if r.category else "",
            f"Price: {r.price}" if r.price else "",
            f"How it works: {r.how_it_works}" if r.how_it_works else "",
            f"Named competitors: {r.competitors}" if r.competitors else "",
        ] if p
    )


def _competitors(r: StudyRequest) -> list[str]:
    return [c.strip() for c in r.competitors.replace(";", ",").split(",") if c.strip()]


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _run_stream(req: StudyRequest):
    """Iterate the graph on a worker thread; yield SSE frames as events arrive."""
    s = get_settings()
    graph = build_graph()
    q: queue.Queue = queue.Queue()

    inp = {
        "raw_product": _brief(req),
        "request_notes": req.notes,
        "competitors": _competitors(req),
        "company": {
            "name": req.company_name, "industry": req.industry, "does": req.company_does,
        },
    }
    if req.panel_size:
        inp["panel_size"] = req.panel_size

    def worker():
        try:
            final = None
            for mode, chunk in graph.stream(
                inp,
                config={"max_concurrency": s.max_concurrency, "recursion_limit": 60},
                stream_mode=["updates", "custom", "values"],
            ):
                if mode == "custom":
                    q.put(("event", chunk))
                elif mode == "updates":
                    for node in chunk:
                        q.put(("node_done", {"node": node}))
                elif mode == "values":
                    final = chunk
            q.put(("done", _final_payload(final)))
        except Exception as exc:  # surface it to the client instead of hanging
            q.put(("error", {"message": str(exc)}))
        finally:
            q.put((None, None))

    threading.Thread(target=worker, daemon=True).start()

    yield _sse("meta", {"library": len(library.index()), "workers": [
        "intake", "research", "recruit_panel", "run_panel", "moderator", "analyze",
    ]})
    while True:
        kind, payload = q.get()
        if kind is None:
            break
        yield _sse(kind, payload)


def _final_payload(state) -> dict:
    if not state or not state.get("report"):
        return {"ok": False}
    return {
        "ok": True,
        "stimulus": state["stimulus"].model_dump(),
        "segments": [s.model_dump() for s in state.get("segments", [])],
        "recruitment_note": state.get("recruitment_note", ""),
        "personas": [p.model_dump() for p in state["personas"]],
        "responses": [r.model_dump() for r in state["responses"]],
        "moderator_notes": state.get("moderator_notes", ""),
        "moderator_probes": state.get("moderator_probes", []),
        "market_evidence": state.get("market_evidence", []),
        "report": state["report"].model_dump(),
        "study_dir": state.get("study_dir", ""),
        "supervisor_log": state.get("supervisor_log", []),
    }


@app.post("/api/study")
def study(req: StudyRequest) -> StreamingResponse:
    return StreamingResponse(_run_stream(req), media_type="text/event-stream")


def main() -> None:
    import uvicorn

    print("RentACrowd UI  →  http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
