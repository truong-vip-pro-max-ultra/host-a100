"""Round-trip tests for anthropic_bridge.buffered_stream (Hướng A: live prose +
buffer-on-tool-call). Run: python3 tests/test_buffered_stream.py

No external deps — feeds synthetic OpenAI SSE into buffered_stream and asserts the
emitted Anthropic SSE. Exits non-zero on any failure."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services import anthropic_bridge as ab  # noqa: E402

TYPES = {"Write": {"path": "string", "content": "string"},
         "Read": {"path": "string"}, "Count": {"n": "integer"}}


class FakeResp:
    """Iterates like an http.client response: yields one byte line at a time."""
    def __init__(self, lines):
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)


class FakeConn:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _oai_line(delta=None, finish=None, usage=None):
    obj = {"choices": [{"index": 0, "delta": delta or {},
                        "finish_reason": finish}]}
    if usage:
        obj["usage"] = usage
    return ("data: " + json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def content_lines(pieces, finish="stop"):
    lines = [_oai_line({"content": p}) for p in pieces]
    lines.append(_oai_line({}, finish=finish))
    lines.append(b"data: [DONE]\n")
    return lines


def run(lines, types=TYPES):
    conn = FakeConn()
    out = b"".join(ab.buffered_stream(FakeResp(lines), conn, "m", 5, types))
    assert conn.closed, "conn was not closed"
    return parse_events(out)


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


def analyze(events):
    texts, tool_partials, tools, stop = {}, {}, [], None
    text_leaks = []
    for ev, d in events:
        if ev == "content_block_start":
            i, cb = d["index"], d["content_block"]
            if cb["type"] == "text":
                texts.setdefault(i, "")
            else:
                tools.append({"index": i, "name": cb["name"]})
                tool_partials[i] = ""
        elif ev == "content_block_delta":
            i, delta = d["index"], d["delta"]
            if delta["type"] == "text_delta":
                texts[i] = texts.get(i, "") + delta["text"]
                if "<tool_call" in delta["text"] or "<function=" in delta["text"]:
                    text_leaks.append(delta["text"])
            elif delta["type"] == "input_json_delta":
                tool_partials[i] += delta["partial_json"]
        elif ev == "message_delta":
            stop = d["delta"]["stop_reason"]
    for t in tools:
        t["input"] = json.loads(tool_partials[t["index"]] or "{}")
    visible = "".join(texts[i] for i in sorted(texts))
    return visible, tools, stop, text_leaks


# --------------------------------------------------------------------------- #
FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def assert_envelope(name, events):
    check(name + ": starts with message_start", events[0][0] == "message_start")
    check(name + ": ping after start", events[1][0] == "ping")
    check(name + ": ends with message_stop", events[-1][0] == "message_stop")


TOOL_BLOCK = ("<tool_call>\n<function=Write>\n<parameter=path>\na.txt\n"
              "</parameter>\n<parameter=content>\nhello world\n</parameter>\n"
              "</function>\n</tool_call>")


def t1_pure_prose():
    print("1) pure prose — streams live, end_turn, no tool")
    ev = run(content_lines(["Xin ", "chào ", "bạn", "!"]))
    visible, tools, stop, leaks = analyze(ev)
    assert_envelope("t1", ev)
    check("t1: text reconstructs", visible == "Xin chào bạn!", repr(visible))
    check("t1: no tools", tools == [])
    check("t1: stop end_turn", stop == "end_turn", stop)
    check("t1: no leaks", not leaks, str(leaks))


def t2_prose_then_tool():
    print("2) prose then tool_call — leading prose live, tool_use parsed")
    ev = run(content_lines(["Để mình tạo file.\n", TOOL_BLOCK], finish="stop"))
    visible, tools, stop, leaks = analyze(ev)
    assert_envelope("t2", ev)
    # Streamed live, so the newline before <tool_call> can't be stripped after
    # the fact (the non-stream path's .strip() does remove it) — keeping it is
    # faithful and harmless.
    check("t2: leading prose kept", visible == "Để mình tạo file.\n", repr(visible))
    check("t2: one tool", len(tools) == 1, str(tools))
    check("t2: tool name Write", tools and tools[0]["name"] == "Write")
    check("t2: tool input",
          tools and tools[0]["input"] == {"path": "a.txt", "content": "hello world"},
          str(tools and tools[0]["input"]))
    check("t2: stop tool_use", stop == "tool_use", stop)
    check("t2: no leaks", not leaks, str(leaks))


def t3_tool_only():
    print("3) tool_call only — no prose, tool_use, stop tool_use")
    block = ("<tool_call>\n<function=Read>\n<parameter=path>\nb.txt\n"
             "</parameter>\n</function>\n</tool_call>")
    ev = run(content_lines([block]))
    visible, tools, stop, leaks = analyze(ev)
    assert_envelope("t3", ev)
    check("t3: no visible text", visible == "", repr(visible))
    check("t3: Read tool", tools and tools[0]["name"] == "Read", str(tools))
    check("t3: input path b.txt", tools and tools[0]["input"] == {"path": "b.txt"})
    check("t3: stop tool_use", stop == "tool_use", stop)
    check("t3: no leaks", not leaks)


def t4_marker_split():
    print("4) <tool_call> split across chunks — caught, not leaked")
    block_rest = ("_call>\n<function=Read>\n<parameter=path>\nc.txt\n"
                  "</parameter>\n</function>\n</tool_call>")
    ev = run(content_lines(["Hi ", "<tool", block_rest]))
    visible, tools, stop, leaks = analyze(ev)
    check("t4: visible is just 'Hi '", visible == "Hi ", repr(visible))
    check("t4: tool parsed", tools and tools[0]["input"] == {"path": "c.txt"}, str(tools))
    check("t4: stop tool_use", stop == "tool_use", stop)
    check("t4: NO marker leak", not leaks, str(leaks))


def t5_false_marker():
    print("5) '<' in prose but no real tool — flushed fully, end_turn")
    ev = run(content_lines(["a < b and c ", "<= d, done"]))
    visible, tools, stop, leaks = analyze(ev)
    check("t5: full text intact", visible == "a < b and c <= d, done", repr(visible))
    check("t5: no tools", tools == [])
    check("t5: stop end_turn", stop == "end_turn", stop)
    # stream ends with a marker-prefix that never completes → must still flush it
    ev2 = run(content_lines(["result <tool"]))
    vis2, _, stop2, leaks2 = analyze(ev2)
    check("t5b: dangling '<tool' flushed", vis2 == "result <tool", repr(vis2))
    check("t5b: no leaks (incomplete marker is plain text)", not leaks2)


def t6_truncated_tool():
    print("6) truncated mid tool (finish=length) — drop tool, stop max_tokens")
    cut = ("Đang ghi file...\n<tool_call>\n<function=Write>\n<parameter=path>\n"
           "big.txt\n</parameter>\n<parameter=content>\nstart of a huge file that "
           "got cut off")  # no closing tags
    ev = run(content_lines([cut], finish="length"))
    visible, tools, stop, leaks = analyze(ev)
    check("t6: leading prose kept", "Đang ghi file..." in visible, repr(visible))
    check("t6: NO tool_use (corrupt dropped)", tools == [], str(tools))
    check("t6: stop max_tokens", stop == "max_tokens", stop)
    check("t6: no leaks", not leaks, str(leaks))


def t7_trailing_text():
    print("7) text after tool_call — emitted as its own block")
    block = ("<tool_call>\n<function=Read>\n<parameter=path>\nx\n</parameter>\n"
             "</function>\n</tool_call>\nĐã đọc xong.")
    ev = run(content_lines([block]))
    visible, tools, stop, leaks = analyze(ev)
    check("t7: tool present", tools and tools[0]["name"] == "Read", str(tools))
    check("t7: trailing text shown", "Đã đọc xong." in visible, repr(visible))
    check("t7: stop tool_use", stop == "tool_use", stop)
    check("t7: no leaks", not leaks)


def t8_native_tool_calls():
    print("8) native OpenAI tool_calls fragments — switch + parse")
    lines = [
        _oai_line({"content": "Ok "}),
        _oai_line({"tool_calls": [{"index": 0, "id": "call_1",
                                   "function": {"name": "Count", "arguments": "{\"n\":"}}]}),
        _oai_line({"tool_calls": [{"index": 0, "function": {"arguments": " 42}"}}]}),
        _oai_line({}, finish="tool_calls"),
        b"data: [DONE]\n",
    ]
    ev = run(lines)
    visible, tools, stop, leaks = analyze(ev)
    check("t8: leading prose live", visible == "Ok ", repr(visible))
    check("t8: native tool parsed", tools and tools[0]["name"] == "Count", str(tools))
    check("t8: int coerced", tools and tools[0]["input"] == {"n": 42}, str(tools))
    check("t8: stop tool_use", stop == "tool_use", stop)
    check("t8: no leaks", not leaks)


def t9_ping_cadence():
    print("9) ping cadence on a slow stream (monotonic mocked to advance)")
    real = ab.time.monotonic
    ticks = iter([0] + [100 * i for i in range(1, 50)])  # each poll +100s

    ab.time.monotonic = lambda: next(ticks)
    try:
        ev = run(content_lines(["one ", "two ", "three ", "four"]))
    finally:
        ab.time.monotonic = real
    pings = sum(1 for e, _ in ev if e == "ping")
    check("t9: extra pings emitted while streaming", pings >= 3, f"pings={pings}")
    visible, _, stop, _ = analyze(ev)
    check("t9: text still intact", visible == "one two three four", repr(visible))


if __name__ == "__main__":
    for fn in (t1_pure_prose, t2_prose_then_tool, t3_tool_only, t4_marker_split,
               t5_false_marker, t6_truncated_tool, t7_trailing_text,
               t8_native_tool_calls, t9_ping_cadence):
        fn()
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        sys.exit(1)
    print("ALL TESTS PASSED")
