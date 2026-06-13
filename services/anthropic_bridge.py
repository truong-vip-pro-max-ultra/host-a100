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
import re
import secrets

# Claude Code likes to ask for very large max_tokens (tens of thousands). This
# clamps only the *generation* budget (OUTPUT tokens) for one turn — it is NOT
# the context window (that's the server's n_ctx, input+output). The cap exists so
# a huge requested max_tokens can't, alone, overflow a small-n_ctx server. Our
# live server runs n_ctx=32768, so 16384 lets a single reply write a large file
# while still leaving ~16k for the input prompt+history. Raise toward n_ctx if
# you bump n_ctx; lower it if a server has a small context window.
_MAX_TOKENS_CEILING = 16384


def _id(prefix):
    return f"{prefix}_{secrets.token_hex(12)}"


def _sse(event, obj):
    """One Anthropic SSE event (named event + JSON data), as bytes."""
    return (f"event: {event}\n"
            f"data: {json.dumps(obj, ensure_ascii=False)}\n\n").encode("utf-8")


# --------------------------------------------------------------------------- #
# Qwen3-Coder native tool-instruction prompt (request side)
# --------------------------------------------------------------------------- #
# CRITICAL: the plain `llama_cpp.server --chat_format chatml` engine does NOT
# inject the OpenAI `tools` array into the prompt. So if we just forward `tools`
# upstream, Qwen3-Coder never learns the tools exist and improvises pseudo-code
# (`write_file(...)`, `echo > a.txt`) instead of its native <tool_call> format —
# and parse_qwen_tool_calls() finds nothing to recover. The native `llama-server`
# binary's --jinja would render the GGUF's own tool template, but this cluster
# can't build it (no CUDA toolkit). So we reproduce Qwen3-Coder's OWN tool
# template here, in pure Python, and fold it into the system prompt. The model
# then emits real <tool_call> blocks, which the response side already parses.
#
# We also render prior assistant tool_use / user tool_result turns BACK into
# Qwen's text format (<tool_call>… / <tool_response>…), because the chatml engine
# ignores the OpenAI `tool_calls`/`tool` role too — the conversation history must
# live entirely in `content`.
_TOOL_FORMAT_INSTRUCTIONS = (
    "\n\nIf you choose to call a function ONLY reply in the following format with "
    "NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n"
    "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
    "that can span\nmultiple lines\n</parameter>\n"
    "</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within <tool_call></tool_call> "
    "XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning before the tool call in plain text\n"
    "- If there is no function call available, answer the question like normal with "
    "your current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>"
)


def _qwen_tool_param(pname, schema):
    schema = schema if isinstance(schema, dict) else {}
    out = [f"\n<parameter>\n<name>{pname}</name>"]
    if schema.get("type"):
        out.append(f"\n<type>{schema['type']}</type>")
    if schema.get("description"):
        out.append(f"\n<description>{schema['description']}</description>")
    out.append("\n</parameter>")
    return "".join(out)


def qwen_tools_prompt(tools):
    """Render Anthropic `tools` into Qwen3-Coder's native tool-instruction block,
    or "" when there are no tools. Faithful to the model's own chat template so
    Qwen reliably emits <tool_call> blocks."""
    funcs = []
    for t in tools or []:
        name = t.get("name")
        if not name:
            continue
        props = (t.get("input_schema") or {}).get("properties") or {}
        params = "".join(_qwen_tool_param(p, s) for p, s in props.items())
        funcs.append(f"\n<function>\n<name>{name}</name>"
                     f"\n<description>{t.get('description', '') or ''}</description>"
                     f"\n<parameters>{params}\n</parameters>\n</function>")
    if not funcs:
        return ""
    return ("# Tools\n\nYou have access to the following functions:\n\n<tools>"
            + "".join(funcs) + "\n</tools>" + _TOOL_FORMAT_INSTRUCTIONS)


def _render_tool_call_text(name, inp):
    """A single assistant tool call as Qwen native <tool_call> text (for history)."""
    out = ["<tool_call>", f"<function={name}>"]
    for k, v in (inp or {}).items():
        if not isinstance(v, str):
            v = json.dumps(v, ensure_ascii=False)
        out.append(f"<parameter={k}>\n{v}\n</parameter>")
    out.append("</function>")
    out.append("</tool_call>")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Request: Anthropic -> OpenAI
# --------------------------------------------------------------------------- #
def to_openai_request(a, served_name):
    """Translate an Anthropic Messages request dict into an OpenAI chat request.

    Tools are injected into the system prompt in Qwen3-Coder's native format (see
    qwen_tools_prompt) rather than sent as the OpenAI `tools` array, because the
    chatml engine ignores that array.
    """
    msgs = []

    system = a.get("system")
    if isinstance(system, list):
        system = "\n".join(b.get("text", "") for b in system
                           if isinstance(b, dict) and b.get("type") == "text")
    system = system or ""
    tools_prompt = qwen_tools_prompt(a.get("tools"))
    if tools_prompt:
        system = (system + "\n\n" + tools_prompt) if system else tools_prompt
    if system:
        msgs.append({"role": "system", "content": system})

    for m in a.get("messages", []):
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue

        text_parts, tool_call_texts, tool_responses = [], [], []
        for blk in content or []:
            if not isinstance(blk, dict):
                continue
            bt = blk.get("type")
            if bt == "text":
                text_parts.append(blk.get("text", ""))
            elif bt == "tool_use":                      # assistant called a tool
                tool_call_texts.append(
                    _render_tool_call_text(blk.get("name"), blk.get("input")))
            elif bt == "tool_result":                   # user returned tool output
                tc = blk.get("content")
                if isinstance(tc, list):
                    tc = "\n".join(x.get("text", "") for x in tc
                                   if isinstance(x, dict) and x.get("type") == "text")
                if not isinstance(tc, str):
                    tc = json.dumps(tc, ensure_ascii=False)
                tool_responses.append(tc)
            # image / other blocks: skipped (text-only upstream model)

        if role == "assistant":
            body = "\n".join([p for p in text_parts if p] + tool_call_texts)
            msgs.append({"role": "assistant", "content": body})
        else:                                            # user
            pieces = [p for p in text_parts if p]
            pieces += [f"<tool_response>\n{r}\n</tool_response>"
                       for r in tool_responses]
            msgs.append({"role": "user", "content": "\n".join(pieces)})

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
    return o


# --------------------------------------------------------------------------- #
# Qwen3-Coder native tool-call parsing
# --------------------------------------------------------------------------- #
# The plain `llama_cpp.server` engine does NOT parse Qwen3-Coder's native
# tool-call format into OpenAI `tool_calls` — the model emits it as raw text in
# `content` and the OpenAI `tool_calls` field stays empty, so Claude Code only
# ever sees a text block and never edits a file. We recover the calls here.
#
# Qwen3-Coder's format (XML-ish, possibly several blocks for parallel calls):
#     <tool_call>
#     <function=Write>
#     <parameter=file_path>
#     a.txt
#     </parameter>
#     <parameter=content>
#     hello
#     </parameter>
#     </function>
#     </tool_call>
_TOOLCALL_RE = re.compile(r"<tool_call>\s*(.*?)</tool_call>", re.DOTALL)
_FUNC_NAME_RE = re.compile(r"<function\s*=\s*([^>\n]+?)\s*>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter\s*=\s*([^>\n]+?)\s*>(.*?)</parameter>",
                       re.DOTALL)


def tool_param_types(tools):
    """Map {tool_name: {param_name: json_schema_type}} from Anthropic tools, so a
    parsed parameter value (always raw text from the model) can be coerced to the
    type the tool actually expects. Returns {} when there are no tools."""
    out = {}
    for t in tools or []:
        name = t.get("name")
        if not name:
            continue
        props = (t.get("input_schema") or {}).get("properties") or {}
        types = {}
        for pname, schema in props.items():
            if isinstance(schema, dict):
                types[pname] = schema.get("type")
        out[name] = types
    return out


def _coerce_param(value, ptype):
    """Coerce a raw <parameter> string to the schema type. A string-typed param
    (e.g. file content that happens to look like JSON) is kept verbatim; numbers/
    booleans/objects/arrays are parsed. Unknown type → best-effort JSON, else str."""
    if ptype == "string":
        return value
    s = value.strip()
    if s == "":
        return value
    if ptype in ("number", "integer"):
        try:
            return json.loads(s)
        except (ValueError, TypeError):
            return value
    if ptype == "boolean":
        low = s.lower()
        return True if low == "true" else False if low == "false" else value
    if ptype in ("object", "array"):
        try:
            return json.loads(s)
        except (ValueError, TypeError):
            return value
    # Unknown/absent schema type: accept JSON only if it's NOT a bare string.
    try:
        parsed = json.loads(s)
    except (ValueError, TypeError):
        return value
    return value if isinstance(parsed, str) else parsed


def parse_qwen_tool_calls(text, types=None):
    """Pull Qwen native <tool_call> blocks out of `text`. Returns
    (clean_text, calls) where calls is [{"name", "input"}] and clean_text is the
    text with the tool-call blocks removed. No blocks → (text, [])."""
    if not text or "<tool_call>" not in text:
        return text, []
    types = types or {}
    calls = []
    for block in _TOOLCALL_RE.findall(text):
        m = _FUNC_NAME_RE.search(block)
        if not m:
            continue
        name = m.group(1).strip()
        ptypes = types.get(name, {})
        inp = {}
        for pname, pval in _PARAM_RE.findall(block):
            pname = pname.strip()
            # Strip only the single framing newline the template adds on each
            # side, so real leading indentation / blank lines survive.
            if pval.startswith("\n"):
                pval = pval[1:]
            if pval.endswith("\n"):
                pval = pval[:-1]
            inp[pname] = _coerce_param(pval, ptypes.get(pname))
        calls.append({"name": name, "input": inp})
    clean = _TOOLCALL_RE.sub("", text).strip()
    return clean, calls


# --------------------------------------------------------------------------- #
# Non-streaming response: OpenAI -> Anthropic
# --------------------------------------------------------------------------- #
_STOP_MAP = {"stop": "end_turn", "length": "max_tokens",
             "tool_calls": "tool_use", "content_filter": "end_turn"}


def openai_response_to_anthropic(o, model, types=None):
    choice = (o.get("choices") or [{}])[0]
    msg = choice.get("message", {}) or {}
    blocks = []
    native_calls = msg.get("tool_calls") or []
    text = msg.get("content") or ""

    # Engine didn't surface tool_calls but the model wrote Qwen's native
    # <tool_call> as text → recover them so Claude Code sees real tool_use.
    synthesized = []
    if not native_calls and "<tool_call>" in text:
        text, synthesized = parse_qwen_tool_calls(text, types)

    if text:
        blocks.append({"type": "text", "text": text})
    for tc in native_calls:
        fn = tc.get("function", {}) or {}
        try:
            inp = json.loads(fn.get("arguments") or "{}")
        except (ValueError, TypeError):
            inp = {}
        blocks.append({"type": "tool_use", "id": tc.get("id") or _id("toolu"),
                       "name": fn.get("name"), "input": inp})
    for c in synthesized:
        blocks.append({"type": "tool_use", "id": _id("toolu"),
                       "name": c["name"], "input": c["input"]})
    if not blocks:
        blocks.append({"type": "text", "text": ""})

    if synthesized:
        stop_reason = "tool_use"
    else:
        stop_reason = _STOP_MAP.get(choice.get("finish_reason"), "end_turn")

    usage = o.get("usage", {}) or {}
    return {
        "id": o.get("id") or _id("msg"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": stop_reason,
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


def sse_from_message(message):
    """Replay a COMPLETE Anthropic message dict (the output of
    openai_response_to_anthropic) as the Anthropic SSE event sequence.

    Used when the client asked for stream=true but we had to read the whole
    upstream reply first to parse Qwen's text tool-calls — we still hand the
    client a well-formed stream (message_start → per-block events → message_stop),
    just not token-by-token. For an agentic client, a correct tool_use block
    matters far more than live tokens.
    """
    usage = message.get("usage") or {}
    start_msg = {"id": message.get("id") or _id("msg"), "type": "message",
                 "role": "assistant", "model": message.get("model"),
                 "content": [], "stop_reason": None, "stop_sequence": None,
                 "usage": {"input_tokens": usage.get("input_tokens", 0),
                           "output_tokens": 0}}
    yield _sse("message_start", {"type": "message_start", "message": start_msg})
    yield _sse("ping", {"type": "ping"})

    for i, blk in enumerate(message.get("content") or []):
        if blk.get("type") == "tool_use":
            yield _sse("content_block_start",
                       {"type": "content_block_start", "index": i,
                        "content_block": {"type": "tool_use",
                                          "id": blk.get("id") or _id("toolu"),
                                          "name": blk.get("name") or "",
                                          "input": {}}})
            partial = json.dumps(blk.get("input") or {}, ensure_ascii=False)
            yield _sse("content_block_delta",
                       {"type": "content_block_delta", "index": i,
                        "delta": {"type": "input_json_delta",
                                  "partial_json": partial}})
        else:
            yield _sse("content_block_start",
                       {"type": "content_block_start", "index": i,
                        "content_block": {"type": "text", "text": ""}})
            yield _sse("content_block_delta",
                       {"type": "content_block_delta", "index": i,
                        "delta": {"type": "text_delta",
                                  "text": blk.get("text", "")}})
        yield _sse("content_block_stop",
                   {"type": "content_block_stop", "index": i})

    yield _sse("message_delta",
               {"type": "message_delta",
                "delta": {"stop_reason": message.get("stop_reason") or "end_turn",
                          "stop_sequence": None},
                "usage": {"output_tokens": usage.get("output_tokens", 0)}})
    yield _sse("message_stop", {"type": "message_stop"})


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
