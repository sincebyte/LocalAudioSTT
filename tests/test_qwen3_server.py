import importlib.util
import json
import sys
import threading
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = ROOT / "server"


def load_cleanup():
    if str(SERVER_DIR) not in sys.path:
        sys.path.insert(0, str(SERVER_DIR))
    spec = importlib.util.spec_from_file_location("cleanup", SERVER_DIR / "cleanup.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cleanup"] = module
    spec.loader.exec_module(module)
    return module


def load_server_module():
    if str(SERVER_DIR) not in sys.path:
        sys.path.insert(0, str(SERVER_DIR))
    spec = importlib.util.spec_from_file_location(
        "qwen3_asr_server", SERVER_DIR / "qwen3_asr_server.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["qwen3_asr_server"] = module
    spec.loader.exec_module(module)
    return module


cleanup = load_cleanup()
server = load_server_module()

# Mirror of the default prompt set by start-server.sh.
DEFAULT_PROMPT = (
    "语音转写：说话人只说中文和英文，以中文为主。结合上下文纠正同音错别字，"
    "人名、地名尽量准确。注意：若听到指令词“发送”或“clear”，把它作为独立的"
    "指令单独成句，并在指令词处触发断句。"
)


# ---------------- cleanup unit tests ----------------


def test_strip_markers():
    s = cleanup.strip_markers
    assert s("language English<asr_text>Hello world.") == "Hello world."
    assert s("language Chinese<asr_text>你好。") == "你好。"
    assert s("no marker here") == "no marker here"


def test_has_spoken_content():
    c = cleanup.has_spoken_content
    assert not c("")
    assert not c("！")
    assert not c("。。。\n？？")
    assert c("你好。")
    assert c("hello")
    assert c("3")


def test_pause_newlines_are_preserved():
    f = cleanup.finalize_transcription
    # The engine breaks a line at each pause; the pipeline passes those breaks
    # through untouched - paragraph layout is the downstream LLM's job.
    assert (
        f("这一段\n说的是同一个意思。\n后面还有一句", organizer="rule")
        == "这一段\n说的是同一个意思。\n后面还有一句"
    )
    assert f("用 API\n做转写", organizer="rule") == "用 API\n做转写"
    # The final ender is the pause artifact and is dropped (see trailing-punct
    # tests); line breaks and in-segment marks stay.
    assert f("Hello World.\nNext line.", organizer="rule") == "Hello World.\nNext line"


def test_enumeration_markers_are_not_rewritten():
    f = cleanup.finalize_transcription
    # No line-initial ordinal rewriting either; text comes back as spoken.
    assert (
        f("嗯，第一点买牛奶\n呃，第二点买鸡蛋", organizer="rule")
        == "第一点买牛奶\n第2点买鸡蛋"
    )
    assert f("一、买牛奶\n二、买鸡蛋", organizer="rule") == "一、买牛奶\n二、买鸡蛋"


def test_none_organizer_keeps_raw_words():
    f = cleanup.finalize_transcription
    assert f("第一点买牛奶", organizer="none") == "第一点买牛奶"


def test_no_trailing_newline_is_added():
    f = cleanup.finalize_transcription
    # Line breaks are never added or merged by the pipeline; only the pause
    # ender at the very end is dropped.
    assert f("你好。", organizer="rule") == "你好"
    assert f("", organizer="rule") == ""


def test_filler_and_glued_filler_removed_in_rule():
    f = cleanup.finalize_transcription
    assert f("那官方的呃飞猪云很好用", organizer="rule") == "那官方的飞猪云很好用"
    assert f("版本啊嗯而且我们要改", organizer="rule") == "版本，而且我们要改"


def test_punctuation_only_returns_empty():
    f = cleanup.finalize_transcription
    assert f("！", organizer="rule") == ""
    assert f("language Chinese<asr_text>。", organizer="rule") == ""
    assert f("谢谢！", organizer="rule") == "谢谢"


def test_trailing_pause_punctuation_is_stripped_by_default():
    f = cleanup.finalize_transcription
    assert f("你好。", organizer="rule") == "你好"
    assert f("真的吗？", organizer="rule") == "真的吗"
    assert f("no way!", organizer="none") == "no way"
    # A run of enders goes together.
    assert f("太好了！！", organizer="rule") == "太好了"
    # A wrapping closing quote is kept; only the ender inside is cut.
    assert f("他说“你好。”", organizer="rule") == "他说“你好”"
    # In-segment punctuation and line breaks are untouched.
    assert f("第一句。\n第二句！", organizer="rule") == "第一句。\n第二句"


def test_trailing_punctuation_strip_can_be_disabled():
    f = cleanup.finalize_transcription
    assert f("你好。", organizer="rule", strip_trailing_punct=False) == "你好。"
    assert (
        f("Hello World.\nNext line.", organizer="rule", strip_trailing_punct=False)
        == "Hello World.\nNext line."
    )


def test_marker_prefix_is_stripped_before_pipeline():
    f = cleanup.finalize_transcription
    assert f("language English<asr_text>clear the screen", organizer="rule") == "clear the screen"


def test_prompt_echo_is_dropped():
    f = cleanup.finalize_transcription
    assert f(DEFAULT_PROMPT, organizer="rule", prompt=DEFAULT_PROMPT) == ""
    assert (
        f("language Chinese<asr_text>" + DEFAULT_PROMPT, organizer="rule", prompt=DEFAULT_PROMPT)
        == ""
    )


def test_real_speech_is_kept_despite_prompt():
    f = cleanup.finalize_transcription
    assert f("你好今天天气不错", organizer="rule", prompt=DEFAULT_PROMPT) == "你好今天天气不错"


def test_empty_prompt_disables_echo_guard():
    f = cleanup.finalize_transcription
    # strip_trailing_punct off here so the test isolates echo-guard behavior.
    assert (
        f(DEFAULT_PROMPT, organizer="rule", prompt="", strip_trailing_punct=False)
        == DEFAULT_PROMPT
    )


@pytest.mark.skipif(importlib.util.find_spec("cn2an") is None, reason="cn2an not installed")
def test_chinese_numerals_become_arabic():
    f = cleanup.finalize_transcription
    assert f("明天下午三点开会", organizer="rule") == "明天下午3点开会"
    assert f("买了三十五份材料", organizer="rule") == "买了35份材料"
    assert f("百分之三十", organizer="none") == "30%"
    assert f("二〇二五年三月五号", organizer="none") == "2025年3月5号"


@pytest.mark.skipif(importlib.util.find_spec("cn2an") is None, reason="cn2an not installed")
def test_acronym_digit_is_joined():
    f = cleanup.finalize_transcription
    # The engine spells an acronym+digit ("MP4") out as letters + a Chinese
    # numeral; the fragments collapse to the Arabic token.
    assert f("上传的是 M P 四文件", organizer="rule") == "上传的是 MP4文件"
    assert f("MP 四 和 MP 三", organizer="rule") == "MP4 和 MP3"
    assert f("这个是 MP四文件", organizer="rule") == "这个是 MP4文件"
    # 一 is a function-word prefix, not the digit 1: API 一下 must stay put.
    assert f("用 API 一下这个接口", organizer="rule") == "用 API 一下这个接口"


@pytest.mark.skipif(importlib.util.find_spec("cn2an") is None, reason="cn2an not installed")
def test_classifier_kuai_keeps_chinese_digit():
    f = cleanup.finalize_transcription
    # 块 is a classifier here, not currency -> 一 stays Chinese.
    assert f("这一块地方", organizer="none") == "这一块地方"
    assert f("两块石头", organizer="none") == "两块石头"
    # Explicit money and composite amounts still digitize.
    assert f("一块钱", organizer="none") == "1块钱"
    assert f("三十五块", organizer="none") == "35块"


def test_stutter_character_is_collapsed():
    f = cleanup.finalize_transcription
    assert (
        f("中台和上云 API 的这这一块也是可以变的呀", organizer="rule")
        == "中台和上云 API 的这一块也是可以变的呀"
    )
    assert f("我我觉得有有问题", organizer="rule") == "我觉得有问题"
    # Same character split by a pause the model punctuated: 这、这、，...
    # An utterance-initial stutter + pause is a filler and leaves entirely.
    assert (
        f("这、这、，我觉得这个它是一块儿内容呀。", organizer="rule")
        == "我觉得这个它是一块儿内容呀"
    )
    # Mid-sentence the repeat collapses to one, keeping a single pause.
    assert f("他说这、这、对", organizer="rule") == "他说这，对"
    # The model may punctuate the stutter with a sentence ender; the echoed
    # character starts a longer word, so it collapses (可。可能 -> 可能) while a
    # genuine sentence pair (这。这。) is left intact.
    assert f("进可。可能靠前", organizer="rule") == "进可能靠前"
    assert f("这个可可能明显是重复", organizer="rule") == "这个可能明显是重复"
    # A genuine sentence pair survives, minus the trailing pause ender.
    assert f("这。这。", organizer="rule") == "这。这"


def test_legitimate_reduplication_is_kept():
    f = cleanup.finalize_transcription
    assert f("妈妈看看刚刚买的书", organizer="rule") == "妈妈看看刚刚买的书"
    # Fixed names that merely contain 可可 survive the stutter pass.
    assert f("可可西里的风景", organizer="rule") == "可可西里的风景"
    assert f("可可豆很好吃", organizer="rule") == "可可豆很好吃"


def test_command_words_are_canonicalized():
    f = cleanup.finalize_transcription
    # Model heard them wrong -> mapped back to the exact command word.
    assert f("发松", organizer="rule") == "发送"
    assert f("法送一下", organizer="rule") == "发送一下"
    assert f("可丽儿", organizer="rule") == "clear"
    assert f("claer", organizer="rule") == "clear"
    assert f("Claire", organizer="rule") == "clear"
    assert f("please C L E A R now", organizer="rule") == "please clear now"
    # General vocabulary stays untouched (no org/emacs mapping by design).
    assert f("奥格猫抖的 org mode 配置", organizer="rule") == "奥格猫抖的 org mode 配置"


# ---------------- HTTP integration with a fake engine ----------------


class _FakeEngineHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.server.last_body = self.rfile.read(length)
        body = json.dumps({"text": self.server.raw}, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _FakeEngineServer(ThreadingHTTPServer):
    def __init__(self, raw: str):
        super().__init__(("127.0.0.1", 0), _FakeEngineHandler)
        self.raw = raw
        self.last_body = b""


def _post_wav(port: int):
    boundary = "----qwen3-test-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="s.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
        "RIFFfake\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="model"\r\n\r\n'
        "qwen3-asr-1.7b\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))["text"]


def _start_engine(raw: str) -> _FakeEngineServer:
    engine = _FakeEngineServer(raw)
    thread = threading.Thread(target=engine.serve_forever, daemon=True)
    thread.start()
    return engine


def test_prompt_is_forwarded_to_engine_when_configured():
    engine = _start_engine("language Chinese<asr_text>好")
    cfg = server.ServerConfig(
        engine_url=f"http://127.0.0.1:{engine.server_port}/v1",
        prompt="MARKER123 你是语音转写校对助手",
    )
    httpd = server.create_server("127.0.0.1", 0, cfg)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        _post_wav(httpd.server_port)
        body = engine.last_body.decode("utf-8", "replace")
        assert 'name="prompt"' in body
        assert "MARKER123 你是语音转写校对助手" in body
    finally:
        httpd.shutdown()
        httpd.server_close()
        engine.shutdown()
        engine.server_close()


def test_transcription_endpoint_runs_full_pipeline():
    engine = _start_engine("language Chinese<asr_text>嗯，第一点买牛奶\n第二点买鸡蛋")
    cfg = server.ServerConfig(engine_url=f"http://127.0.0.1:{engine.server_port}/v1")
    httpd = server.create_server("127.0.0.1", 0, cfg)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        text = _post_wav(httpd.server_port)
        assert text == "第一点买牛奶\n第2点买鸡蛋"
    finally:
        httpd.shutdown()
        httpd.server_close()
        engine.shutdown()
        engine.server_close()


def test_transcription_endpoint_returns_empty_for_punctuation_only():
    engine = _start_engine("language Chinese<asr_text>。")
    cfg = server.ServerConfig(engine_url=f"http://127.0.0.1:{engine.server_port}/v1")
    httpd = server.create_server("127.0.0.1", 0, cfg)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        text = _post_wav(httpd.server_port)
        assert text == ""
    finally:
        httpd.shutdown()
        httpd.server_close()
        engine.shutdown()
        engine.server_close()


def test_transcription_endpoint_drops_prompt_echo():
    engine = _start_engine(DEFAULT_PROMPT)
    cfg = server.ServerConfig(
        engine_url=f"http://127.0.0.1:{engine.server_port}/v1",
        prompt=DEFAULT_PROMPT,
    )
    httpd = server.create_server("127.0.0.1", 0, cfg)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        text = _post_wav(httpd.server_port)
        assert text == ""
    finally:
        httpd.shutdown()
        httpd.server_close()
        engine.shutdown()
        engine.server_close()


def test_transcription_endpoint_strips_trailing_punct_by_default():
    engine = _start_engine("language Chinese<asr_text>你好。")
    cfg = server.ServerConfig(engine_url=f"http://127.0.0.1:{engine.server_port}/v1")
    httpd = server.create_server("127.0.0.1", 0, cfg)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        text = _post_wav(httpd.server_port)
        assert text == "你好"
    finally:
        httpd.shutdown()
        httpd.server_close()
        engine.shutdown()
        engine.server_close()


def test_transcription_endpoint_can_keep_trailing_punct():
    engine = _start_engine("language Chinese<asr_text>你好。")
    cfg = server.ServerConfig(
        engine_url=f"http://127.0.0.1:{engine.server_port}/v1",
        strip_trailing_punct=False,
    )
    httpd = server.create_server("127.0.0.1", 0, cfg)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        text = _post_wav(httpd.server_port)
        assert text == "你好。"
    finally:
        httpd.shutdown()
        httpd.server_close()
        engine.shutdown()
        engine.server_close()


# ---------------- streaming endpoint ----------------

_MARKER = ["language", " Chinese", "<asr_text>"]


class _FakeStreamEngineHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.server.last_body = self.rfile.read(length)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.end_headers()
        for piece in self.server.pieces:
            chunk = {"choices": [{"index": 0, "delta": {"content": piece}}]}
            line = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            self.wfile.write(line.encode("utf-8"))
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


class _FakeStreamEngineServer(ThreadingHTTPServer):
    def __init__(self, pieces):
        super().__init__(("127.0.0.1", 0), _FakeStreamEngineHandler)
        self.pieces = pieces
        self.last_body = b""


def _start_stream_engine(pieces):
    engine = _FakeStreamEngineServer(pieces)
    thread = threading.Thread(target=engine.serve_forever, daemon=True)
    thread.start()
    return engine


def _post_stream_wav(port: int):
    boundary = "----qwen3-test-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="s.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
        "RIFFfake\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/audio/transcriptions/stream",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    deltas = []
    final = None
    with urllib.request.urlopen(request, timeout=10) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event.get("type") == "delta":
                deltas.append(event["text"])
            elif event.get("type") == "final":
                final = event["text"]
    return "".join(deltas), final


def test_marker_stream_filter_handles_split_tag():
    f = server.MarkerStreamFilter()
    assert f.feed("language") == ""
    assert f.feed(" Chinese") == ""
    out = f.feed("<asr_text>你好") + f.feed("世界")
    assert out == "你好世界"
    assert f.flush() == ""


def test_marker_stream_filter_passes_text_without_marker():
    f = server.MarkerStreamFilter()
    assert f.feed("没有") == ""
    assert f.feed("标记") == ""
    assert f.flush() == "没有标记"


def test_stream_endpoint_emits_deltas_then_cleaned_final():
    pieces = _MARKER + ["嗯，", "第一点买牛奶", "。"]
    engine = _start_stream_engine(pieces)
    cfg = server.ServerConfig(engine_url=f"http://127.0.0.1:{engine.server_port}/v1")
    httpd = server.create_server("127.0.0.1", 0, cfg)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        deltas, final = _post_stream_wav(httpd.server_port)
        # Deltas are the raw decode (marker removed) for live display...
        assert deltas == "嗯，第一点买牛奶。"
        # ...while final is the cleaned text the caller keeps.
        assert final == "第一点买牛奶"
        payload = json.loads(engine.last_body.decode("utf-8"))
        assert payload["stream"] is True
        assert payload["messages"][0]["content"][0]["type"] == "input_audio"
    finally:
        httpd.shutdown()
        httpd.server_close()
        engine.shutdown()
        engine.server_close()
