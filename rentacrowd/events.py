"""Progress events for the live UI.

Nodes call `emit(...)` to narrate what they are doing - which sub-agent they
spawned, whether the persona cache was hit, how many LLM calls they made. The
Streamlit UI consumes these via LangGraph's `stream_mode="custom"` and renders
them under each node.

`emit` is a no-op when there is no active graph run (e.g. unit tests, or a plain
`graph.invoke` without custom streaming), so nodes can call it unconditionally.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from langgraph.config import get_stream_writer

EventKind = Literal[
    "start",      # node began
    "route",      # supervisor chose the next worker
    "cache",      # persona library reuse / miss
    "spawn",      # a sub-agent / parallel worker was created
    "llm",        # an LLM call was made
    "compute",    # deterministic (Python) work, no LLM
    "result",     # node produced its output
]


def emit(node: str, kind: EventKind, message: str, **data: Any) -> None:
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return  # not inside a graph run
    writer(
        {
            "node": node,
            "kind": kind,
            "message": message,
            "data": data,
            "ts": time.time(),
        }
    )
