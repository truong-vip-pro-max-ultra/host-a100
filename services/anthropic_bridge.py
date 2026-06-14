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
import time

# Claude Code likes to ask for very large max_tokens (tens of thousands). This
# clamps only the *generation* budget (OUTPUT tokens) for one turn — it is NOT
# the context window (that's the server's n_ctx, input+output). The cap exists so
# a huge requested max_tokens can't, alone, overflow a small-n_ctx server. The
# live server runs n_ctx=65536 (L40S 48GB; Qwen3-Coder's tiny GQA KV cache makes
# 64k ≈ 6 GiB), so 32768 lets a single reply write a large file in one turn while
# still leaving ~32k for the input prompt+history. Raise toward n_ctx if you bump
# n_ctx; lower it if a server has a small context window. History: 8192 → 16384 →
# 24576 → 32768 (2026-06-14, paired with the n_ctx 32768→65536 bump) because large
# file writes were truncating mid-<tool_call> → "Error writing file".
_MAX_TOKENS_CEILING = 32768

# When we buffer a reply and replay it as a synthesized SSE (sse_from_message),
# emit the text in slices this many CHARACTERS wide instead of one huge delta —
# a single giant text_delta makes Claude Code's TUI duplicate/mis-paint the text.
_TEXT_CHUNK = 24


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


def _qwen_tool_param(pname, schema, required=False):
    schema = schema if isinstance(schema, dict) else {}
    out = [f"\n<parameter>\n<name>{pname}</name>"]
    if schema.get("type"):
        out.append(f"\n<type>{schema['type']}</type>")
    if schema.get("description"):
        out.append(f"\n<description>{schema['description']}</description>")
    # Tell the model which params are mandatory. Without this Qwen guesses and
    # omits required args (e.g. the Agent/Task tool's `description`), so Claude
    # Code rejects the call with "Invalid tool parameters" and the model has to
    # retry — each retry costs another full-prompt prefill.
    out.append(f"\n<required>{'true' if required else 'false'}</required>")
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
        schema = t.get("input_schema") or {}
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        params = "".join(_qwen_tool_param(p, s, p in required)
                         for p, s in props.items())
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
def _anthropic_tools_to_openai(tools):
    """Anthropic `tools` -> OpenAI function tools. The native llama-server
    (`--jinja`) renders these into the GGUF's own tool template and parses the
    model's calls back into real `tool_calls`, so we forward them as the OpenAI
    `tools` array instead of injecting a hand-rolled prompt (qwen_tools_prompt)."""
    out = []
    for t in tools or []:
        name = t.get("name")
        if not name:
            continue
        out.append({"type": "function", "function": {
            "name": name,
            "description": t.get("description", "") or "",
            "parameters": t.get("input_schema") or {"type": "object",
                                                     "properties": {}}}})
    return out


def _map_tool_choice(tc):
    """Anthropic tool_choice -> OpenAI tool_choice. None when unset/unknown."""
    if not isinstance(tc, dict):
        return None
    t = tc.get("type")
    if t == "auto":
        return "auto"
    if t == "any":
        return "required"
    if t == "none":
        return "none"
    if t == "tool" and tc.get("name"):
        return {"type": "function", "function": {"name": tc["name"]}}
    return None


def _tool_result_text(blk):
    """Flatten an Anthropic tool_result block's content into a string."""
    tc = blk.get("content")
    if isinstance(tc, list):
        tc = "\n".join(x.get("text", "") for x in tc
                       if isinstance(x, dict) and x.get("type") == "text")
    if not isinstance(tc, str):
        tc = json.dumps(tc, ensure_ascii=False)
    return tc


def to_openai_request(a, served_name):
    """Translate an Anthropic Messages request into an OpenAI chat request for the
    native llama-server engine.

    Tools are forwarded as the OpenAI `tools` array (llama-server's `--jinja`
    renders + parses them natively into real `tool_calls`), and tool history is
    expressed with `tool_calls` / `tool`-role messages — NOT injected as a Qwen
    text template or folded into `<tool_call>` text. (The old text-injection path
    was for the chatml `llamacpp` engine, which ignored the `tools` array.)
    """
    msgs = []

    system = a.get("system")
    if isinstance(system, list):
        system = "\n".join(b.get("text", "") for b in system
                           if isinstance(b, dict) and b.get("type") == "text")
    system = system or ""
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
            elif bt == "tool_use":                       # assistant called a tool
                tool_calls.append({
                    "id": blk.get("id") or _id("call"),
                    "type": "function",
                    "function": {
                        "name": blk.get("name") or "",
                        "arguments": json.dumps(blk.get("input") or {},
                                                ensure_ascii=False)}})
            elif bt == "tool_result":                    # user returned tool output
                tool_results.append((blk.get("tool_use_id"),
                                     _tool_result_text(blk)))
            # image / other blocks: skipped (text-only upstream model)

        text = "\n".join(p for p in text_parts if p)
        if role == "assistant":
            msg = {"role": "assistant", "content": text or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            msgs.append(msg)
        else:                                            # user
            # Tool results become `tool` messages (must follow the assistant
            # tool_calls turn); any plain user text follows as a user message.
            for tid, c in tool_results:
                tm = {"role": "tool", "content": c}
                if tid:
                    tm["tool_call_id"] = tid
                msgs.append(tm)
            if text:
                msgs.append({"role": "user", "content": text})

    o = {"model": served_name, "messages": msgs}

    tools = _anthropic_tools_to_openai(a.get("tools"))
    if tools:
        o["tools"] = tools
        tc = _map_tool_choice(a.get("tool_choice"))
        if tc is not None:
            o["tool_choice"] = tc

    mt = a.get("max_tokens")
    if mt:
        o["max_tokens"] = min(int(mt), _MAX_TOKENS_CEILING)
    if a.get("temperature") is not None:
        o["temperature"] = a["temperature"]
    if a.get("top_p") is not None:
        o["top_p"] = a["top_p"]
    if a.get("stop_sequences"):
        o["stop"] = list(a["stop_sequences"])
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
# Match a block up to its closing </tool_call> OR, if Qwen left it unclosed
# (it sometimes emits a bare trailing `<tool_call>` or gets cut mid-call), up to
# end of string — so a dangling open tag is still consumed instead of leaking
# into the visible text.
_TOOLCALL_RE = re.compile(r"<tool_call>\s*(.*?)(?:</tool_call>|\Z)", re.DOTALL)
_FUNC_NAME_RE = re.compile(r"<function\s*=\s*([^>\n]+?)\s*>", re.DOTALL)
# Close a <parameter> on its own </parameter>, OR on the start of the next
# <parameter>/</function>, OR at end-of-string. The last two let us still recover
# a param whose closing tag was lost because generation was TRUNCATED mid-content
# (hit the output-token ceiling while writing a big file) or the model simply
# forgot to close it. Non-greedy, so a complete param still closes on its own tag.
_PARAM_RE = re.compile(
    r"<parameter\s*=\s*([^>\n]+?)\s*>(.*?)(?:</parameter>|(?=<parameter\b)|(?=</function\b)|\Z)",
    re.DOTALL)
# Stray framing tags left over after block removal (e.g. an orphan </tool_call>).
_ORPHAN_TAG_RE = re.compile(r"</?(?:tool_call|function|parameter)\b[^>]*>")


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
    if not text or "tool_call" not in text:
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
    clean = _TOOLCALL_RE.sub("", text)
    clean = _ORPHAN_TAG_RE.sub("", clean).strip()
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
    if not native_calls and "tool_call" in text:
        text, synthesized = parse_qwen_tool_calls(text, types)

    # A tool call recovered from a TRUNCATED reply (model ran into the output
    # ceiling mid-<parameter=content>) is incomplete — emitting it would make
    # Claude Code write a corrupt/partial file ("Error writing file" loop). Drop
    # it and report max_tokens instead, so the client knows the turn was cut and
    # continues rather than running a broken Write.
    truncated = choice.get("finish_reason") == "length"
    if truncated and synthesized:
        synthesized = []

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
    for ev in _message_body_events(message):
        yield ev


def _message_body_events(message):
    """The Anthropic SSE events AFTER message_start/ping: the indexed content
    blocks, then message_delta + message_stop. Shared by sse_from_message and
    buffered_stream (which sends its own message_start up front)."""
    usage = message.get("usage") or {}
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
            # Emit the text in small slices, not one giant text_delta. The real
            # Anthropic stream sends many tiny deltas; Claude Code's TUI renderer
            # mis-paints (duplicates) a single huge delta. Slicing by code point
            # (not bytes) keeps UTF-8 / Vietnamese diacritics intact.
            text = blk.get("text", "")
            for j in range(0, len(text), _TEXT_CHUNK):
                yield _sse("content_block_delta",
                           {"type": "content_block_delta", "index": i,
                            "delta": {"type": "text_delta",
                                      "text": text[j:j + _TEXT_CHUNK]}})
        yield _sse("content_block_stop",
                   {"type": "content_block_stop", "index": i})

    yield _sse("message_delta",
               {"type": "message_delta",
                "delta": {"stop_reason": message.get("stop_reason") or "end_turn",
                          "stop_sequence": None},
                "usage": {"output_tokens": usage.get("output_tokens", 0)}})
    yield _sse("message_stop", {"type": "message_stop"})


# Heartbeat cadence while buffering a tool-bearing reply. Claude Code's request
# rides a Cloudflare tunnel with a ~120s read timeout (Error 524): if no bytes
# reach the client for that long, the tunnel kills the connection. A long file
# write can generate for minutes, so we emit a `ping` at least this often.
_PING_INTERVAL = 15  # seconds


# A tool-bearing reply starts buffering the moment Qwen's native tool-call
# opener appears in the text. Until then the prose is streamed LIVE (like the
# Chat tab) so an explanatory turn shows tokens immediately instead of only
# after the whole generation finishes. `_HOLDBACK` is how many trailing chars we
# keep un-emitted so a `<tool_call>` split across two SSE chunks is still caught
# by the substring search before any of it leaks out as visible text.
_TOOLCALL_OPEN = "<tool_call>"
_HOLDBACK = len(_TOOLCALL_OPEN) - 1


def buffered_stream(resp, conn, model, input_tokens, types=None):
    """Stream the LEADING prose live, then buffer once a tool call appears.

    Claude Code always sends tools, but most turns are still mostly prose
    (explanations, plans) with a tool call only at the end — or none at all. We
    forward that prose to the client as real text deltas as it generates, and
    only switch to buffering when Qwen's `<tool_call>` opener (or a native
    tool_calls fragment) shows up. From that point we accumulate the rest, parse
    it with the same logic as the non-stream path, and replay the tool_use (plus
    any trailing text) blocks. This gives Chat-tab latency for the talky part
    while still producing correct tool_use for the agentic part.

    We never emit a partial `<tool_call>` as text: a holdback tail guards a
    marker split across SSE chunks, and a complete marker is found by substring
    search before the text ahead of it is flushed. A long pure-tool generation
    still emits a `ping` every `_PING_INTERVAL`s so the Cloudflare tunnel's ~120s
    read timeout (Error 524) never trips. `resp` must be a STREAMING OpenAI
    response (stream=true). Closes `conn` when exhausted."""
    msg_id = _id("msg")
    start_msg = {"id": msg_id, "type": "message", "role": "assistant",
                 "model": model, "content": [], "stop_reason": None,
                 "stop_sequence": None,
                 "usage": {"input_tokens": input_tokens, "output_tokens": 0}}
    yield _sse("message_start", {"type": "message_start", "message": start_msg})
    yield _sse("ping", {"type": "ping"})

    state = {"index": -1, "text_open": False}

    def _open_text():
        state["index"] += 1
        state["text_open"] = True
        return _sse("content_block_start",
                    {"type": "content_block_start", "index": state["index"],
                     "content_block": {"type": "text", "text": ""}})

    def _close_block():
        state["text_open"] = False
        return _sse("content_block_stop",
                    {"type": "content_block_stop", "index": state["index"]})

    def _emit_text(s):
        """Yield `s` as text deltas, opening a text block if needed and slicing to
        _TEXT_CHUNK so Claude Code's TUI doesn't mis-paint one giant delta."""
        if not s:
            return
        if not state["text_open"]:
            yield _open_text()
        for j in range(0, len(s), _TEXT_CHUNK):
            yield _sse("content_block_delta",
                       {"type": "content_block_delta", "index": state["index"],
                        "delta": {"type": "text_delta", "text": s[j:j + _TEXT_CHUNK]}})

    pending = ""             # live prose not yet flushed (holds the marker tail)
    buffering = False        # True once a tool call (text or native) has begun
    tail_parts = []          # all content accumulated AFTER the switch
    tool_calls = {}          # native fragmented tool_calls: index -> slot
    live_chars = 0
    finish_reason = None
    usage = {}
    last_ping = time.monotonic()

    try:
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
            choice = (obj.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            if obj.get("usage"):
                usage = obj["usage"]
            native = delta.get("tool_calls") or []
            piece = delta.get("content") or ""

            # A native tool_calls fragment means tools have begun → flush any safe
            # pending prose, close the live block, and switch to buffering.
            if native and not buffering:
                if pending:
                    for ev in _emit_text(pending):
                        yield ev
                    live_chars += len(pending)
                    pending = ""
                if state["text_open"]:
                    yield _close_block()
                buffering = True

            if not buffering and piece:
                pending += piece
                idx = pending.find(_TOOLCALL_OPEN)
                if idx != -1:
                    # Complete marker: everything before it is safe prose.
                    safe, rest = pending[:idx], pending[idx:]
                    for ev in _emit_text(safe):
                        yield ev
                    live_chars += len(safe)
                    if state["text_open"]:
                        yield _close_block()
                    buffering = True
                    tail_parts.append(rest)
                    pending = ""
                elif len(pending) > _HOLDBACK:
                    # No marker yet: flush all but a holdback tail that could be
                    # the start of a `<tool_call>` split across chunks.
                    safe, pending = pending[:-_HOLDBACK], pending[-_HOLDBACK:]
                    for ev in _emit_text(safe):
                        yield ev
                    live_chars += len(safe)
            elif buffering:
                if piece:
                    tail_parts.append(piece)
                for tc in native:
                    slot = tool_calls.setdefault(
                        tc.get("index", 0), {"id": None, "name": None, "args": []})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["args"].append(fn["arguments"])

            now = time.monotonic()
            if now - last_ping >= _PING_INTERVAL:
                last_ping = now
                yield _sse("ping", {"type": "ping"})
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    # ----- finalize ---------------------------------------------------------- #
    if not buffering:
        # Pure prose: flush the leftover holdback tail (plain text, no marker).
        for ev in _emit_text(pending):
            yield ev
        live_chars += len(pending)
        if state["index"] == -1:          # nothing at all → one empty text block
            yield _open_text()
        if state["text_open"]:
            yield _close_block()
        out_tokens = usage.get("completion_tokens") or max(1, live_chars // 4)
        stop = _STOP_MAP.get(finish_reason, "end_turn")
        yield _sse("message_delta",
                   {"type": "message_delta",
                    "delta": {"stop_reason": stop, "stop_sequence": None},
                    "usage": {"output_tokens": out_tokens}})
        yield _sse("message_stop", {"type": "message_stop"})
        return

    # Buffering happened: parse the tail for tool calls + trailing text via the
    # SAME conversion the non-stream path uses (handles truncation-drop, native
    # tool_calls, type coercion), then emit the remaining blocks continuing the
    # block index (the leading prose was already streamed as its own block).
    tail_text = "".join(tail_parts)
    tail_msg = {"role": "assistant", "content": tail_text or None}
    if tool_calls:
        tail_msg["tool_calls"] = [
            {"id": slot["id"], "type": "function",
             "function": {"name": slot["name"], "arguments": "".join(slot["args"])}}
            for _, slot in sorted(tool_calls.items())]
    oai_resp = {"id": msg_id, "usage": usage,
                "choices": [{"message": tail_msg, "finish_reason": finish_reason}]}
    message = openai_response_to_anthropic(oai_resp, model, types)

    for blk in message.get("content") or []:
        if blk.get("type") == "tool_use":
            state["index"] += 1
            yield _sse("content_block_start",
                       {"type": "content_block_start", "index": state["index"],
                        "content_block": {"type": "tool_use",
                                          "id": blk.get("id") or _id("toolu"),
                                          "name": blk.get("name") or "", "input": {}}})
            partial = json.dumps(blk.get("input") or {}, ensure_ascii=False)
            yield _sse("content_block_delta",
                       {"type": "content_block_delta", "index": state["index"],
                        "delta": {"type": "input_json_delta", "partial_json": partial}})
            yield _sse("content_block_stop",
                       {"type": "content_block_stop", "index": state["index"]})
        else:
            text = blk.get("text", "")
            # Skip the empty filler block openai_response_to_anthropic adds when a
            # reply has no blocks — but only if we already emitted something, so a
            # tool-only-then-stripped reply still yields a valid (empty) block.
            if text == "" and state["index"] >= 0:
                continue
            for ev in _emit_text(text):
                yield ev
            if state["text_open"]:
                yield _close_block()

    if state["index"] == -1:              # nothing emitted at all → empty block
        yield _open_text()
        yield _close_block()

    total = live_chars + len(tail_text)
    out_tokens = usage.get("completion_tokens") or max(1, total // 4)
    yield _sse("message_delta",
               {"type": "message_delta",
                "delta": {"stop_reason": message.get("stop_reason") or "end_turn",
                          "stop_sequence": None},
                "usage": {"output_tokens": out_tokens}})
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
