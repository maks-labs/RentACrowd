"""NIM-backed chat model + prompt-based structured output.

Everything in RentACrowd talks to NVIDIA NIM through `panel_llm()` / `analysis_llm()`.
A single process-wide rate limiter is shared by every model instance so the
combined request rate across all parallel persona workers stays under the NIM
free-tier ceiling (~40 req/min).

The nemotron models on the free tier are "thinking" models: they reason in plain
prose (not <think> tags) and reject server-side structured output (`guided_json`,
tool calling). So `structured_call` does structured output the only way that
survives them: one user turn, no system message, a compact *example* object (not
a JSON Schema), a generous token budget so the trailing JSON is never truncated,
then a tolerant parse + validate with a self-correcting retry.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from rentacrowd.config import get_settings

_T = TypeVar("_T", bound=BaseModel)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@lru_cache
def _rate_limiter() -> InMemoryRateLimiter:
    s = get_settings()
    return InMemoryRateLimiter(
        requests_per_second=s.requests_per_second,
        check_every_n_seconds=0.1,
        max_bucket_size=s.requests_per_minute,
    )


def _make(model: str, temperature: float) -> ChatNVIDIA:
    s = get_settings()
    return ChatNVIDIA(
        model=model,
        base_url=s.nim_base_url,
        api_key=s.nvidia_api_key,
        temperature=temperature,
        max_completion_tokens=s.max_output_tokens,
        timeout=s.request_timeout,
        rate_limiter=_rate_limiter(),
        # nemotron models reason in plain prose by default, which is slow (30s+)
        # and blows the token budget. This turns thinking OFF at the template level
        # so the model answers immediately.
        model_kwargs={"chat_template_kwargs": {"thinking": False}},
    )


# --------------------------------------------------------------------------- #
# JSON-schema -> compact example object (what we actually show the model)
# --------------------------------------------------------------------------- #


def _example_from_schema(schema: dict, defs: dict | None = None) -> Any:
    defs = defs or schema.get("$defs", {})

    if "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        return _example_from_schema(defs[name], defs)
    if "anyOf" in schema:
        return _example_from_schema(schema["anyOf"][0], defs)
    if "enum" in schema:
        return schema["enum"][0]

    t = schema.get("type")
    if t == "object":
        return {
            k: _example_from_schema(v, defs)
            for k, v in schema.get("properties", {}).items()
        }
    if t == "array":
        return [_example_from_schema(schema.get("items", {}), defs)]
    if t == "integer":
        return 0
    if t == "number":
        return 0.0
    if t == "boolean":
        return False
    return schema.get("description", "string")[:60] or "string"


def _example_json(schema: type[BaseModel]) -> str:
    return json.dumps(_example_from_schema(schema.model_json_schema()), indent=1)


def _extract_json(text: str) -> str:
    text = _THINK.sub("", text).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    end = text.rfind("}")
    if end > start:
        return text[start : end + 1]
    return _repair_truncated(text[start:])


def _repair_truncated(s: str) -> str:
    """Best-effort close of a JSON object the model got cut off mid-emit.

    Walk the text tracking string state and bracket depth; stop at the last
    position that could be a valid value boundary, then append the missing
    closers. Try json.loads while trimming back until it parses.
    """
    import json as _json

    depth: list[str] = []
    instr = esc = False
    safe = 0  # last index right after a completed value / at a clean boundary
    for i, ch in enumerate(s):
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
                safe = i + 1
            continue
        if ch == '"':
            instr = True
        elif ch in "{[":
            depth.append(ch)
        elif ch in "}]":
            if depth:
                depth.pop()
            safe = i + 1
        elif ch in "0123456789eE.+-" or ch in "truefalsn":
            safe = i + 1
        elif ch in " \t\n\r,:":
            pass

    head = s[:safe].rstrip().rstrip(",")
    for _ in range(len(depth) + 1):
        cand = head + "".join("}" if b == "{" else "]" for b in reversed(depth))
        cand = re.sub(r",(\s*[}\]])", r"\1", cand)
        try:
            _json.loads(cand)
            return cand
        except Exception:
            head = re.sub(r",?\s*(\"[^\"]*\"\s*:\s*)?[^,{}\[\]]*$", "", head).rstrip().rstrip(",")
    return cand


class _Retryable(Exception):
    pass


def structured_call(llm: ChatNVIDIA, schema: type[_T], instruction: str) -> _T:
    """Ask `llm` to satisfy `instruction` and return it parsed into `schema`."""
    example = _example_json(schema)
    prompt = (
        f"{instruction.strip()}\n\n"
        "Reply with ONLY one JSON object using exactly these keys (same shape, "
        "realistic values). No preamble, no explanation, no code fence:\n"
        f"{example}"
    )
    convo: list[BaseMessage] = [HumanMessage(prompt)]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=20),
        retry=retry_if_exception_type(
            (_Retryable, ConnectionError, TimeoutError, OSError)
        ),
        reraise=True,
    )
    def _run() -> _T:
        raw = llm.invoke(convo).content
        raw = raw if isinstance(raw, str) else str(raw)
        try:
            return schema.model_validate(json.loads(_extract_json(raw)))
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            convo.append(
                HumanMessage(
                    f"That output was invalid: {e}\nReturn only the corrected JSON object."
                )
            )
            raise _Retryable(str(e)) from e

    return _run()


@lru_cache
def panel_llm() -> ChatNVIDIA:
    """Model used for the bulk of the work: persona generation + panel responses."""
    return _make(get_settings().panel_model, temperature=0.9)


@lru_cache
def analysis_llm() -> ChatNVIDIA:
    """Model for the moderator debrief and the final roll-up. Low temperature."""
    return _make(get_settings().analysis_model, temperature=0.3)
