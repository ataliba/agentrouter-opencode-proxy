#!/usr/bin/env python3
"""
agentrouter-proxy: thin reverse-proxy for agentrouter.org.

AgentRouter's Aliyun WAF fingerprints TLS handshakes AND inspects
Anthropic SDK-specific headers (x-stainless-*, user-agent). Raw httpx
is blocked, and the AsyncAnthropic client is also rejected because its
asyncio SSL implementation produces a different TLS fingerprint.
Only requests made through the Python sync `anthropic` SDK pass the check.

This proxy keeps a single sync `anthropic.Anthropic` client (shared
connection pool, short keepalive_expiry to prevent zombie connections
when the upstream load balancer silently closes idle sockets).
Both the non-streaming and streaming paths offload to a thread via
asyncio.to_thread / a thread+queue so the FastAPI event loop stays free.

Usage:
    python proxy.py           # reads key from ~/.config/opencode/api_keys/AGENT_ROUTER_API_KEY
    AGENTROUTER_API_KEY=sk-... python proxy.py
"""
import asyncio
import json
import os
import queue
import threading
import time
import uuid
from pathlib import Path

import anthropic
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

# ── Config ────────────────────────────────────────────────────────────────────

TARGET   = "https://agentrouter.org"
HOST     = os.environ.get("HOST", "127.0.0.1")
PORT     = 7187
KEY_FILE = Path.home() / ".config/opencode/api_keys/AGENT_ROUTER_API_KEY"

# Seconds between received SSE chunks before aborting a streaming request.
CHUNK_TIMEOUT = 120

# agentrouter.org's load balancer drops idle connections after ~30 s.
# Setting keepalive_expiry below that threshold prevents our pool from
# trying to reuse a connection the server has already closed (zombie socket).
_HTTP_CLIENT = httpx.Client(
    limits=httpx.Limits(
        max_connections=50,
        max_keepalive_connections=5,
        keepalive_expiry=20.0,
    ),
    timeout=httpx.Timeout(connect=10, read=CHUNK_TIMEOUT, write=10, pool=5),
)


def _api_key() -> str:
    k = os.environ.get("AGENTROUTER_API_KEY", "").strip()
    if not k and KEY_FILE.exists():
        k = KEY_FILE.read_text().strip()
    if not k:
        raise RuntimeError(
            "No AGENTROUTER_API_KEY. "
            f"Set env var or create {KEY_FILE}"
        )
    return k


_anthropic_client: anthropic.Anthropic | None = None


def _client() -> anthropic.Anthropic:
    """Return the module-level sync client singleton (connection pool reused across requests)."""
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(
            api_key=_api_key(),
            base_url=TARGET,
            http_client=_HTTP_CLIENT,
        )
    return _anthropic_client


# ── Request translation ───────────────────────────────────────────────────────

_SKIP  = {"stream"}                     # handled separately
_STRIP = {"thinking", "output_config"}  # non-standard; trigger agentrouter content filter


def _kwargs(body: dict) -> dict:
    """Forward all fields except stream (handled separately) and non-standard extras."""
    return {k: v for k, v in body.items() if k not in _SKIP and k not in _STRIP}


# ── Streaming helper ──────────────────────────────────────────────────────────

def _stream_worker(kw: dict, q: queue.Queue) -> None:
    """
    Run inside a thread. Uses the sync Anthropic SDK's with_streaming_response
    to get raw SSE bytes and puts them into the queue, stripping any
    non-standard event types (e.g. billing_summary) that break OpenCode's parser.
    """
    SKIP_EVENTS: set[bytes] = {b"billing_summary"}

    try:
        with _client().messages.with_streaming_response.create(**kw) as resp:
            buf        = b""
            skip_block = False

            for raw_chunk in resp.iter_bytes(chunk_size=1024):
                buf += raw_chunk

                while True:
                    nl = buf.find(b"\n")
                    if nl == -1:
                        break
                    line     = buf[: nl + 1]   # include \n
                    buf      = buf[nl + 1:]
                    stripped = line.rstrip(b"\r\n")

                    if stripped.startswith(b"event:"):
                        event_name = stripped[6:].strip()
                        skip_block = event_name in SKIP_EVENTS
                        if skip_block:
                            continue
                    elif skip_block:
                        if stripped == b"":
                            skip_block = False  # blank line ends the event block
                        continue

                    q.put(line)

            if buf:
                q.put(buf)

    except Exception as exc:
        q.put(exc)
    finally:
        q.put(None)  # sentinel


async def _stream_gen(kw: dict):
    q: queue.Queue = queue.Queue()
    t = threading.Thread(target=_stream_worker, args=(kw, q), daemon=True)
    t.start()
    loop = asyncio.get_running_loop()
    while True:
        try:
            chunk = await asyncio.wait_for(
                loop.run_in_executor(None, q.get),
                timeout=CHUNK_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"No chunk received from agentrouter.org in {CHUNK_TIMEOUT}s — upstream stalled"
            )
        if chunk is None:
            break
        if isinstance(chunk, Exception):
            raise chunk
        yield chunk


# ── OpenAI-compat translation ────────────────────────────────────────────────
# Some clients (e.g. Odysseus) only support the OpenAI "custom URL" shape and
# hit /v1/chat/completions with OpenAI request/response bodies. Anthropic is
# the only format agentrouter.org's WAF accepts (see module docstring), so we
# translate both directions here rather than proxying OpenAI bodies through.

_STOP_REASON = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}


def _openai_to_anthropic_kwargs(body: dict) -> dict:
    system_parts = [m["content"] for m in body.get("messages", []) if m.get("role") == "system"]
    messages = [
        {"role": m["role"], "content": m.get("content", "")}
        for m in body.get("messages", [])
        if m.get("role") in ("user", "assistant")
    ]

    kw = {
        "model": body.get("model", "claude-opus-4-6"),
        "messages": messages,
        "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens") or 4096,
    }
    if system_parts:
        kw["system"] = "\n".join(system_parts)
    if "temperature" in body:
        kw["temperature"] = body["temperature"]
    if "top_p" in body:
        kw["top_p"] = body["top_p"]
    if body.get("stop"):
        stop = body["stop"]
        kw["stop_sequences"] = stop if isinstance(stop, list) else [stop]
    return kw


def _openai_chunk(completion_id: str, created: int, model: str, delta: dict, finish_reason: str | None) -> bytes:
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n".encode()


def _openai_stream_worker(kw: dict, q: queue.Queue, completion_id: str, created: int, model: str) -> None:
    """Runs in a thread: consumes Anthropic SSE events and re-emits them as OpenAI chunks."""
    try:
        sent_role = False
        with _client().messages.stream(**kw) as stream:
            for event in stream:
                if event.type == "content_block_delta" and getattr(event.delta, "type", None) == "text_delta":
                    delta = {"content": event.delta.text}
                    if not sent_role:
                        delta["role"] = "assistant"
                        sent_role = True
                    q.put(_openai_chunk(completion_id, created, model, delta, None))
                elif event.type == "message_delta" and event.delta.stop_reason:
                    finish_reason = _STOP_REASON.get(event.delta.stop_reason, "stop")
                    q.put(_openai_chunk(completion_id, created, model, {}, finish_reason))
        q.put(b"data: [DONE]\n\n")
    except Exception as exc:
        q.put(exc)
    finally:
        q.put(None)


async def _openai_stream_gen(kw: dict, completion_id: str, created: int, model: str):
    q: queue.Queue = queue.Queue()
    t = threading.Thread(
        target=_openai_stream_worker, args=(kw, q, completion_id, created, model), daemon=True
    )
    t.start()
    loop = asyncio.get_running_loop()
    while True:
        try:
            chunk = await asyncio.wait_for(loop.run_in_executor(None, q.get), timeout=CHUNK_TIMEOUT)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"No chunk received from agentrouter.org in {CHUNK_TIMEOUT}s — upstream stalled"
            )
        if chunk is None:
            break
        if isinstance(chunk, Exception):
            raise chunk
        yield chunk


# ── Routes ────────────────────────────────────────────────────────────────────

app = FastAPI()


@app.post("/v1/messages")
@app.post("/messages")
async def messages(request: Request):
    body = await request.json()
    kw   = _kwargs(body)

    if body.get("stream", False):
        kw["stream"] = True

        async def _safe_stream():
            try:
                async for chunk in _stream_gen(kw):
                    yield chunk
            except anthropic.APIStatusError as e:
                err = json.dumps({"type": "error", "error": {"type": "api_error", "message": str(e.body)}})
                yield f"event: error\ndata: {err}\n\n".encode()
            except Exception as e:
                err = json.dumps({"type": "error", "error": {"type": "api_error", "message": str(e)}})
                yield f"event: error\ndata: {err}\n\n".encode()

        return StreamingResponse(
            _safe_stream(),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
        )

    # Non-streaming: sync SDK call in a thread to keep the event loop free
    def _run():
        return _client().messages.create(**kw)

    try:
        msg = await asyncio.to_thread(_run)
        return Response(content=msg.model_dump_json(), media_type="application/json")
    except anthropic.APIStatusError as e:
        return Response(
            content=json.dumps(e.body) if e.body else b"",
            status_code=e.status_code,
            media_type="application/json",
        )
    except Exception as e:
        return Response(
            content=json.dumps({"error": {"message": str(e), "type": "proxy_error"}}),
            status_code=500,
            media_type="application/json",
        )


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    kw   = _openai_to_anthropic_kwargs(body)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model = kw["model"]

    if body.get("stream", False):

        async def _safe_stream():
            try:
                async for chunk in _openai_stream_gen(kw, completion_id, created, model):
                    yield chunk
            except anthropic.APIStatusError as e:
                err = json.dumps({"error": {"message": str(e.body), "type": "api_error"}})
                yield f"data: {err}\n\n".encode()
            except Exception as e:
                err = json.dumps({"error": {"message": str(e), "type": "proxy_error"}})
                yield f"data: {err}\n\n".encode()

        return StreamingResponse(
            _safe_stream(),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
        )

    def _run():
        return _client().messages.create(**kw)

    try:
        msg = await asyncio.to_thread(_run)
    except anthropic.APIStatusError as e:
        return Response(
            content=json.dumps({"error": {"message": str(e.body), "type": "api_error"}}),
            status_code=e.status_code,
            media_type="application/json",
        )
    except Exception as e:
        return Response(
            content=json.dumps({"error": {"message": str(e), "type": "proxy_error"}}),
            status_code=500,
            media_type="application/json",
        )

    text = "".join(block.text for block in msg.content if block.type == "text")
    resp = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": _STOP_REASON.get(msg.stop_reason, "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": msg.usage.input_tokens,
            "completion_tokens": msg.usage.output_tokens,
            "total_tokens": msg.usage.input_tokens + msg.usage.output_tokens,
        },
    }
    return Response(content=json.dumps(resp), media_type="application/json")


@app.get("/v1/models")
@app.get("/models")
async def models():
    """Stub model list — only lists models confirmed working on agentrouter.org."""
    return {
        "object": "list",
        "data": [
            {"id": "claude-opus-4-6", "object": "model"},
        ],
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"AgentRouter proxy  →  {TARGET}")
    print(f"Listening on http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
