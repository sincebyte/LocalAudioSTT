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


def test_rule_organizer_numbers_enumeration():
    f = cleanup.finalize_transcription
    assert f("嗯，第一点买牛奶\n呃，第二点买鸡蛋", organizer="rule") == "1. 买牛奶\n2. 买鸡蛋\n\n"
    assert f("一、买牛奶\n二、买鸡蛋", organizer="rule") == "1. 买牛奶\n2. 买鸡蛋\n\n"


def test_pause_newlines_are_merged_into_one_paragraph():
    f = cleanup.finalize_transcription
    # The engine breaks a line at each pause; the fragments rejoin as one
    # paragraph instead of one line per breath.
    assert (
        f("这一段\n说的是同一个意思。\n后面还有一句", organizer="rule")
        == "这一段说的是同一个意思。后面还有一句\n\n"
    )
    # Mixed CJK/ASCII seams keep their space; English lines rejoin with one.
    assert f("用 API\n做转写", organizer="rule") == "用 API 做转写\n\n"
    assert f("Hello World.\nNext line.", organizer="rule") == "Hello World. Next line.\n\n"


def test_enumeration_continuation_lines_stay_numbered():
    f = cleanup.finalize_transcription
    assert (
        f("第一点买牛奶\n和面包\n第二点买鸡蛋\n还有水果", organizer="rule")
        == "1. 买牛奶和面包\n2. 买鸡蛋还有水果\n\n"
    )


def test_none_organizer_keeps_raw_words():
    f = cleanup.finalize_transcription
    assert f("第一点买牛奶", organizer="none") == "第一点买牛奶\n\n"


def test_trailing_separator_is_configurable():
    f = cleanup.finalize_transcription
    # Default keeps the paragraph blank line; the HTTP server passes "" because
    # OpenChamber dictation joins segments with a space.
    assert f("你好。", organizer="rule") == "你好。\n\n"
    assert f("你好。", organizer="rule", trailing="") == "你好。"
    assert f("", organizer="rule", trailing="") == ""


def test_filler_and_glued_filler_removed_in_rule():
    f = cleanup.finalize_transcription
    assert f("那官方的呃飞猪云很好用", organizer="rule") == "那官方的飞猪云很好用\n\n"
    assert f("版本啊嗯而且我们要改", organizer="rule") == "版本，而且我们要改\n\n"


def test_punctuation_only_returns_empty():
    f = cleanup.finalize_transcription
    assert f("！", organizer="rule") == ""
    assert f("language Chinese<asr_text>。", organizer="rule") == ""
    assert f("谢谢！", organizer="rule") == "谢谢！\n\n"


def test_marker_prefix_is_stripped_before_pipeline():
    f = cleanup.finalize_transcription
    assert f("language English<asr_text>clear the screen", organizer="rule") == "clear the screen\n\n"


def test_prompt_echo_is_dropped():
    f = cleanup.finalize_transcription
    assert f(DEFAULT_PROMPT, organizer="rule", prompt=DEFAULT_PROMPT) == ""
    assert (
        f("language Chinese<asr_text>" + DEFAULT_PROMPT, organizer="rule", prompt=DEFAULT_PROMPT)
        == ""
    )


def test_real_speech_is_kept_despite_prompt():
    f = cleanup.finalize_transcription
    assert f("你好今天天气不错", organizer="rule", prompt=DEFAULT_PROMPT) == "你好今天天气不错\n\n"


def test_empty_prompt_disables_echo_guard():
    f = cleanup.finalize_transcription
    assert f(DEFAULT_PROMPT, organizer="rule", prompt="") == DEFAULT_PROMPT + "\n\n"


@pytest.mark.skipif(importlib.util.find_spec("cn2an") is None, reason="cn2an not installed")
def test_chinese_numerals_become_arabic():
    f = cleanup.finalize_transcription
    assert f("明天下午三点开会", organizer="rule") == "明天下午3点开会\n\n"
    assert f("买了三十五份材料", organizer="rule") == "买了35份材料\n\n"
    assert f("百分之三十", organizer="none") == "30%\n\n"
    assert f("二〇二五年三月五号", organizer="none") == "2025年3月5号\n\n"


@pytest.mark.skipif(importlib.util.find_spec("cn2an") is None, reason="cn2an not installed")
def test_acronym_digit_is_joined():
    f = cleanup.finalize_transcription
    # The engine spells an acronym+digit ("MP4") out as letters + a Chinese
    # numeral; the fragments collapse to the Arabic token.
    assert f("上传的是 M P 四文件", organizer="rule") == "上传的是 MP4文件\n\n"
    assert f("MP 四 和 MP 三", organizer="rule") == "MP4 和 MP3\n\n"
    assert f("这个是 MP四文件", organizer="rule") == "这个是 MP4文件\n\n"
    # 一 is a function-word prefix, not the digit 1: API 一下 must stay put.
    assert f("用 API 一下这个接口", organizer="rule") == "用 API 一下这个接口\n\n"


@pytest.mark.skipif(importlib.util.find_spec("cn2an") is None, reason="cn2an not installed")
def test_classifier_kuai_keeps_chinese_digit():
    f = cleanup.finalize_transcription
    # 块 is a classifier here, not currency -> 一 stays Chinese.
    assert f("这一块地方", organizer="none") == "这一块地方\n\n"
    assert f("两块石头", organizer="none") == "两块石头\n\n"
    # Explicit money and composite amounts still digitize.
    assert f("一块钱", organizer="none") == "1块钱\n\n"
    assert f("三十五块", organizer="none") == "35块\n\n"


def test_stutter_character_is_collapsed():
    f = cleanup.finalize_transcription
    assert (
        f("中台和上云 API 的这这一块也是可以变的呀", organizer="rule")
        == "中台和上云 API 的这一块也是可以变的呀\n\n"
    )
    assert f("我我觉得有有问题", organizer="rule") == "我觉得有问题\n\n"
    # Same character split by a pause the model punctuated: 这、这、，...
    # An utterance-initial stutter + pause is a filler and leaves entirely.
    assert (
        f("这、这、，我觉得这个它是一块儿内容呀。", organizer="rule")
        == "我觉得这个它是一块儿内容呀。\n\n"
    )
    # Mid-sentence the repeat collapses to one, keeping a single pause.
    assert f("他说这、这、对", organizer="rule") == "他说这，对\n\n"
    # The model may punctuate the stutter with a sentence ender; the echoed
    # character starts a longer word, so it collapses (可。可能 -> 可能) while a
    # genuine sentence pair (这。这。) is left intact.
    assert f("进可。可能靠前", organizer="rule") == "进可能靠前\n\n"
    assert f("这个可可能明显是重复", organizer="rule") == "这个可能明显是重复\n\n"
    assert f("这。这。", organizer="rule") == "这。这。\n\n"


def test_legitimate_reduplication_is_kept():
    f = cleanup.finalize_transcription
    assert f("妈妈看看刚刚买的书", organizer="rule") == "妈妈看看刚刚买的书\n\n"
    # Fixed names that merely contain 可可 survive the stutter pass.
    assert f("可可西里的风景", organizer="rule") == "可可西里的风景\n\n"
    assert f("可可豆很好吃", organizer="rule") == "可可豆很好吃\n\n"


def test_command_words_are_canonicalized():
    f = cleanup.finalize_transcription
    # Model heard them wrong -> mapped back to the exact command word.
    assert f("发松", organizer="rule") == "发送\n\n"
    assert f("法送一下", organizer="rule") == "发送一下\n\n"
    assert f("可丽儿", organizer="rule") == "clear\n\n"
    assert f("claer", organizer="rule") == "clear\n\n"
    assert f("Claire", organizer="rule") == "clear\n\n"
    assert f("please C L E A R now", organizer="rule") == "please clear now\n\n"
    # General vocabulary stays untouched (no org/emacs mapping by design).
    assert f("奥格猫抖的 org mode 配置", organizer="rule") == "奥格猫抖的 org mode 配置\n\n"


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
        assert text == "1. 买牛奶\n2. 买鸡蛋"
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
