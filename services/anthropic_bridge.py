"""
Anthropic Messages API <-> OpenAI Chat Completions translation.

Claude Code (and other Anthropic-SDK clients) speak the `/v1/messages` API, but
our GPU servers (llama.cpp) speak the OpenAI `/v1/chat/completions` API. This
module is the bridge: it converts an Anthropic request into an OpenAI request,
and converts the OpenAI reply back into the Anthropic shape — for BOTH the
one-shot JSON response and the streaming SSE event sequence, including tool use
(Claude Code is agentic and relies on tools heavily).

It is intentionally dependency-free (stdlib json/secrets) so it can run inside
the same single-process app. The app.py route owns the HTTP plumbing; this
module owns only the format translation.
"""
import json
import secrets

# Claude Code likes to ask for very large max_tokens (tens of thousands). A
# llama.cpp server with a modest n_ctx would reject/clip that awkwardly, so we
# clamp the *generation* budget to something sane for a chat/coding turn. Raise
# this if you run a server with a big context window.
_MAX_TOKENS_CEILING = 8192


def _id(prefix):
    return f"{prefix}_{secrets.token_hex(12)}"


def _sse(event, obj):
    """One Anthropic SSE event (named event + JSON data), as bytes."""
    return (f"event: {event}\n"
            f"data: {json.dumps(obj, ensure_ascii=False)}\n\n").encode("utf-8")


# --------------------------------------------------------------------------- #
# Request: Anthropic -> OpenAI
# --------------------------------------------------------------------------- #
def to_openai_request(a, served_name):
    """Translate an Anthropic Messages request dict into an OpenAI chat request."""
    msgs = []

    system = a.get("system")
    if isinstance(system, list):
        system = "\n".join(b.get("text", "") for b in system
                           if isinstance(b, dict) and b.get("type") == "text")
    if system:
        msgs.append({"role": "system", "content": system})

    for m in a.get("messages", []):
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue

        text_parts, tool_calls, tool_results = [], [], []
        for blk in content or []:
            if not isinstance(blk, dict):
                continue
            bt = blk.get("type")
            if bt == "text":
                text_parts.append(blk.get("text", ""))
            elif bt == "tool_use":                      # assistant called a tool
                tool_calls.append({
                    "id": blk.get("id") or _id("toolu"),
                    "type": "function",
                    "function": {"name": blk.get("name"),
                                 "arguments": json.dumps(blk.get("input", {}),
                                                         ensure_ascii=False)},
                })
            elif bt == "tool_result":                   # user returned tool output
                tc = blk.get("content")
                if isinstance(tc, list):
                    tc = "\n".join(x.get("text", "") for x in tc
                                   if isinstance(x, dict) and x.get("type") == "text")
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": blk.get("tool_use_id"),
                    "content": tc if isinstance(tc, str)
                    else json.dumps(tc, ensure_ascii=False),
                })
            # image / other blocks: skipped (text-only upstream model)

        if role == "assistant":
            asst = {"role": "assistant",
                    "content": "".join(text_parts) or None}
            if tool_calls:
                asst["tool_calls"] = tool_calls
            msgs.append(asst)
        else:                                            # user
            if text_parts:
                msgs.append({"role": "user", "content": "".join(text_parts)})
            msgs.extend(tool_results)

    o = {"model": served_name, "messages": msgs}

    mt = a.get("max_tokens")
    if mt:
        o["max_tokens"] = min(int(mt), _MAX_TOKENS_CEILING)
    if a.get("temperature") is not None:
        o["temperature"] = a["temperature"]
    if a.get("top_p") is not None:
        o["top_p"] = a["top_p"]
    if a.get("stop_sequences"):
        o["stop"] = a["stop_sequences"]

    if a.get("tools"):
        tools = []
        for t in a["tools"]:
            if not t.get("name"):
                continue
            tools.append({"type": "function", "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            }})
        if tools:
            o["tools"] = tools

    tc = a.get("tool_choice")
    if isinstance(tc, dict):
        ty = tc.get("type")
        if ty == "auto":
            o["tool_choice"] = "auto"
        elif ty == "any":
            o["tool_choice"] = "required"
        elif ty == "tool" and tc.get("name"):
            o["tool_choice"] = {"type": "function",
                                "function": {"name": tc["name"]}}
    return o


# --------------------------------------------------------------------------- #
# Non-streaming response: OpenAI -> Anthropic
# --------------------------------------------------------------------------- #
_STOP_MAP = {"stop": "end_turn", "length": "max_tokens",
             "tool_calls": "tool_use", "content_filter": "end_turn"}


def openai_response_to_anthropic(o, model):
    choice = (o.get("choices") or [{}])[0]
    msg = choice.get("message", {}) or {}
    blocks = []
    if msg.get("content"):
        blocks.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {}) or {}
        try:
            inp = json.loads(fn.get("arguments") or "{}")
        except (ValueError, TypeError):
            inp = {}
        blocks.append({"type": "tool_use", "id": tc.get("id") or _id("toolu"),
                       "name": fn.get("name"), "input": inp})
    if not blocks:
        blocks.append({"type": "text", "text": ""})

    usage = o.get("usage", {}) or {}
    return {
        "id": o.get("id") or _id("msg"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": _STOP_MAP.get(choice.get("finish_reason"), "end_turn"),
        "stop_sequence": None,
        "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                  "output_tokens": usage.get("completion_tokens", 0)},
    }


# --------------------------------------------------------------------------- #
# Streaming response: OpenAI SSE -> Anthropic SSE
# --------------------------------------------------------------------------- #
class _StreamState:
    """Turns the OpenAI delta stream into the Anthropic event stream. Anthropic
    models content as a sequence of indexed blocks (text / tool_use); we open and
    close those blocks as the OpenAI deltas (content vs tool_calls) come in."""

    def __init__(self, model, input_tokens):
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = 0
        self.block_index = -1
        self.cur_type = None              # 'text' | 'tool' | None
        self.tool_slots = {}              # openai tool-call index -> anthropic block index
        self.finish_reason = None

    def start(self):
        msg = {"id": _id("msg"), "type": "message", "role": "assistant",
               "model": self.model, "content": [], "stop_reason": None,
               "stop_sequence": None,
               "usage": {"input_tokens": self.input_tokens, "output_tokens": 0}}
        yield _sse("message_start", {"type": "message_start", "message": msg})
        yield _sse("ping", {"type": "ping"})

    def _close_current(self):
        if self.cur_type is not None:
            self.cur_type = None
            return [_sse("content_block_stop",
                         {"type": "content_block_stop", "index": self.block_index})]
        return []

    def _open_text(self):
        out = self._close_current()
        self.block_index += 1
        self.cur_type = "text"
        out.append(_sse("content_block_start",
                        {"type": "content_block_start", "index": self.block_index,
                         "content_block": {"type": "text", "text": ""}}))
        return out

    def _tool_delta(self, tc):
        out = []
        idx = tc.get("index", 0)
        fn = tc.get("function") or {}
        if idx not in self.tool_slots:
            out += self._close_current()
            self.block_index += 1
            self.cur_type = "tool"
            self.tool_slots[idx] = self.block_index
            out.append(_sse("content_block_start",
                            {"type": "content_block_start", "index": self.block_index,
                             "content_block": {"type": "tool_use",
                                               "id": tc.get("id") or _id("toolu"),
                                               "name": fn.get("name") or "",
                                               "input": {}}}))
        args = fn.get("arguments")
        if args:
            out.append(_sse("content_block_delta",
                            {"type": "content_block_delta",
                             "index": self.tool_slots[idx],
                             "delta": {"type": "input_json_delta",
                                       "partial_json": args}}))
        return out

    def process(self, obj):
        out = []
        usage = obj.get("usage")
        if usage:
            if usage.get("completion_tokens") is not None:
                self.output_tokens = usage["completion_tokens"]
            if usage.get("prompt_tokens") is not None:
                self.input_tokens = usage["prompt_tokens"]
        choices = obj.get("choices") or []
        if not choices:
            return out
        ch = choices[0]
        if ch.get("finish_reason"):
            self.finish_reason = ch["finish_reason"]
        delta = ch.get("delta") or {}
        content = delta.get("content")
        if content:
            if self.cur_type != "text":
                out += self._open_text()
            out.append(_sse("content_block_delta",
                            {"type": "content_block_delta", "index": self.block_index,
                             "delta": {"type": "text_delta", "text": content}}))
            self.output_tokens += max(1, len(content) // 4)
        for tc in delta.get("tool_calls") or []:
            out += self._tool_delta(tc)
        return out

    def finalize(self):
        out = []
        if self.block_index == -1:        # nothing streamed → emit an empty block
            out += self._open_text()
        out += self._close_current()
        stop = _STOP_MAP.get(self.finish_reason, "end_turn")
        out.append(_sse("message_delta",
                        {"type": "message_delta",
                         "delta": {"stop_reason": stop, "stop_sequence": None},
                         "usage": {"output_tokens": self.output_tokens}}))
        out.append(_sse("message_stop", {"type": "message_stop"}))
        return out


def stream(resp, conn, model, input_tokens):
    """Generator yielding Anthropic SSE bytes, driven by an upstream OpenAI SSE
    response (`resp`). Closes `conn` when the stream is exhausted."""
    state = _StreamState(model, input_tokens)
    try:
        for chunk in state.start():
            yield chunk
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except ValueError:
                continue
            for out in state.process(obj):
                yield out
        for out in state.finalize():
            yield out
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Token estimate (for /v1/messages/count_tokens and message_start). Rough — a
# llama.cpp server doesn't expose Anthropic's tokenizer, so ~4 chars/token.
# --------------------------------------------------------------------------- #
def estimate_tokens(a):
    n = 0
    system = a.get("system")
    if isinstance(system, str):
        n += len(system)
    elif isinstance(system, list):
        for b in system:
            if isinstance(b, dict):
                n += len(b.get("text", ""))
    for m in a.get("messages", []):
        c = m.get("content")
        if isinstance(c, str):
            n += len(c)
        elif isinstance(c, list):
            for blk in c:
                if not isinstance(blk, dict):
                    continue
                bt = blk.get("type")
                if bt == "text":
                    n += len(blk.get("text", ""))
                elif bt == "tool_use":
                    n += len(json.dumps(blk.get("input", {})))
                elif bt == "tool_result":
                    tc = blk.get("content")
                    n += len(tc) if isinstance(tc, str) else len(json.dumps(tc))
    return max(1, n // 4)
