"""Tests for the native-tool_calls bridge refactor: to_openai_request forwards
the OpenAI `tools` array + `tool`/`tool_calls` history (no Qwen text injection),
and the streaming/non-streaming response paths turn native tool_calls into
Anthropic tool_use. Run: python3 tests/test_native_tools.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services import anthropic_bridge as ab  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


class FakeResp:
    def __init__(self, lines):
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)


class FakeConn:
    def close(self):
        pass


def _oai_line(delta=None, finish=None, usage=None):
    obj = {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if usage:
        obj["usage"] = usage
    return ("data: " + json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def parse_events(blob):
    events = []
    for chunk in blob.split(b"\n\n"):
        chunk = chunk.decode("utf-8")
        ev = data = None
        for ln in chunk.splitlines():
            if ln.startswith("event: "):
                ev = ln[7:].strip()
            elif ln.startswith("data: "):
                data = json.loads(ln[6:])
        if ev:
            events.append((ev, data))
    return events


# --------------------------------------------------------------------------- #
TOOLS = [{"name": "Read", "description": "read a file",
          "input_schema": {"type": "object",
                           "properties": {"path": {"type": "string"}},
                           "required": ["path"]}}]


def t1_request_shape():
    print("1) to_openai_request: forwards tools + tool_choice, no text injection")
    a = {
        "system": "You are Claude Code.",
        "tools": TOOLS,
        "tool_choice": {"type": "auto"},
        "max_tokens": 100000,
        "messages": [
            {"role": "user", "content": "đọc file a"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Để mình đọc."},
                {"type": "tool_use", "id": "tu_1", "name": "Read",
                 "input": {"path": "a"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1",
                 "content": "noi dung file"}]},
        ],
    }
    o = ab.to_openai_request(a, "qwen")
    msgs = o["messages"]

    check("t1: tools forwarded as OpenAI shape",
          o.get("tools") == [{"type": "function", "function": {
              "name": "Read", "description": "read a file",
              "parameters": TOOLS[0]["input_schema"]}}], json.dumps(o.get("tools")))
    check("t1: tool_choice auto", o.get("tool_choice") == "auto", str(o.get("tool_choice")))
    check("t1: max_tokens clamped to ceiling",
          o["max_tokens"] == ab._MAX_TOKENS_CEILING, str(o.get("max_tokens")))

    sys_msg = msgs[0]
    check("t1: system kept verbatim (no '# Tools' inject)",
          sys_msg["role"] == "system" and sys_msg["content"] == "You are Claude Code.",
          repr(sys_msg["content"]))
    check("t1: no <tool_call> text anywhere",
          not any("<tool_call>" in (m.get("content") or "") for m in msgs))

    asst = next(m for m in msgs if m["role"] == "assistant")
    check("t1: assistant has tool_calls",
          asst.get("tool_calls") and asst["tool_calls"][0]["id"] == "tu_1"
          and asst["tool_calls"][0]["function"]["name"] == "Read", json.dumps(asst))
    check("t1: assistant tool_calls arguments are JSON",
          json.loads(asst["tool_calls"][0]["function"]["arguments"]) == {"path": "a"})
    check("t1: assistant text preserved", asst.get("content") == "Để mình đọc.")

    tool_msg = next(m for m in msgs if m["role"] == "tool")
    check("t1: tool_result -> tool role w/ id + content",
          tool_msg.get("tool_call_id") == "tu_1"
          and tool_msg.get("content") == "noi dung file", json.dumps(tool_msg))


def t2_tool_choice_map():
    print("2) tool_choice mapping")
    mk = lambda tc: ab.to_openai_request({"tools": TOOLS, "tool_choice": tc,
                                          "messages": []}, "q").get("tool_choice")
    check("t2: any -> required", mk({"type": "any"}) == "required")
    check("t2: none -> none", mk({"type": "none"}) == "none")
    check("t2: tool -> function", mk({"type": "tool", "name": "Read"})
          == {"type": "function", "function": {"name": "Read"}})
    check("t2: no tools -> no tool_choice key",
          "tool_choice" not in ab.to_openai_request({"messages": []}, "q"))


def t3_stream_native_toolcall():
    print("3) streaming native tool_calls -> Anthropic tool_use")
    lines = [
        _oai_line({"content": "Mình đọc nhé. "}),
        _oai_line({"tool_calls": [{"index": 0, "id": "call_1",
                                   "function": {"name": "Read", "arguments": ""}}]}),
        _oai_line({"tool_calls": [{"index": 0,
                                   "function": {"arguments": "{\"path\":"}}]}),
        _oai_line({"tool_calls": [{"index": 0,
                                   "function": {"arguments": " \"a.txt\"}"}}]}),
        _oai_line({}, finish="tool_calls", usage={"completion_tokens": 12}),
        b"data: [DONE]\n",
    ]
    out = b"".join(ab.stream(FakeResp(lines), FakeConn(), "q", 5))
    events = parse_events(out)

    texts, tool_partials, tools, stop = {}, {}, [], None
    for ev, d in events:
        if ev == "content_block_start":
            i, cb = d["index"], d["content_block"]
            if cb["type"] == "tool_use":
                tools.append({"index": i, "name": cb["name"]})
                tool_partials[i] = ""
            else:
                texts.setdefault(i, "")
        elif ev == "content_block_delta":
            i, delta = d["index"], d["delta"]
            if delta["type"] == "text_delta":
                texts[i] = texts.get(i, "") + delta["text"]
            elif delta["type"] == "input_json_delta":
                tool_partials[i] += delta["partial_json"]
        elif ev == "message_delta":
            stop = d["delta"]["stop_reason"]

    visible = "".join(texts[i] for i in sorted(texts))
    check("t3: starts message_start", events[0][0] == "message_start")
    check("t3: leading text streamed", visible == "Mình đọc nhé. ", repr(visible))
    check("t3: one tool_use Read", tools and tools[0]["name"] == "Read", str(tools))
    check("t3: arguments reassembled",
          tools and json.loads(tool_partials[tools[0]["index"]]) == {"path": "a.txt"},
          tool_partials)
    check("t3: stop_reason tool_use", stop == "tool_use", stop)
    check("t3: ends message_stop", events[-1][0] == "message_stop")


def t4_nonstream_native_toolcall():
    print("4) non-stream native tool_calls -> tool_use message")
    oai_resp = {"id": "x", "choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": "ok",
        "tool_calls": [{"id": "call_9", "type": "function",
                        "function": {"name": "Read",
                                     "arguments": "{\"path\": \"b.txt\"}"}}]}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    msg = ab.openai_response_to_anthropic(oai_resp, "q")
    blocks = msg["content"]
    tu = [b for b in blocks if b["type"] == "tool_use"]
    check("t4: has tool_use", tu and tu[0]["name"] == "Read", json.dumps(blocks))
    check("t4: input parsed", tu and tu[0]["input"] == {"path": "b.txt"})
    check("t4: stop_reason tool_use", msg["stop_reason"] == "tool_use", msg["stop_reason"])
    check("t4: text block kept", any(b["type"] == "text" and b["text"] == "ok"
                                     for b in blocks))


def t5_plain_text_still_works():
    print("5) plain text request/response (no tools) unaffected")
    o = ab.to_openai_request({"messages": [{"role": "user", "content": "hi"}]}, "q")
    check("t5: no tools key", "tools" not in o)
    check("t5: simple user msg", o["messages"][-1] == {"role": "user", "content": "hi"})
    lines = [_oai_line({"content": "Chào "}), _oai_line({"content": "bạn"}),
             _oai_line({}, finish="stop"), b"data: [DONE]\n"]
    events = parse_events(b"".join(ab.stream(FakeResp(lines), FakeConn(), "q", 1)))
    txt = "".join(d["delta"]["text"] for ev, d in events
                  if ev == "content_block_delta" and d["delta"]["type"] == "text_delta")
    stop = next(d["delta"]["stop_reason"] for ev, d in events if ev == "message_delta")
    check("t5: text streamed", txt == "Chào bạn", repr(txt))
    check("t5: stop end_turn", stop == "end_turn", stop)


if __name__ == "__main__":
    for fn in (t1_request_shape, t2_tool_choice_map, t3_stream_native_toolcall,
               t4_nonstream_native_toolcall, t5_plain_text_still_works):
        fn()
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        sys.exit(1)
    print("ALL TESTS PASSED")
