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


def test_none_organizer_keeps_raw_words():
    f = cleanup.finalize_transcription
    assert f("第一点买牛奶", organizer="none") == "第一点买牛奶\n\n"


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


@pytest.mark.skipif(importlib.util.find_spec("cn2an") is None, reason="cn2an not installed")
def test_chinese_numerals_become_arabic():
    f = cleanup.finalize_transcription
    assert f("明天下午三点开会", organizer="rule") == "明天下午3点开会\n\n"
    assert f("买了三十五份材料", organizer="rule") == "买了35份材料\n\n"
    assert f("百分之三十", organizer="none") == "30%\n\n"
    assert f("二〇二五年三月五号", organizer="none") == "2025年3月5号\n\n"


# ---------------- HTTP integration with a fake engine ----------------


class _FakeEngineHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)  # consume body
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


def test_transcription_endpoint_runs_full_pipeline():
    engine = _start_engine("language Chinese<asr_text>嗯，第一点买牛奶\n第二点买鸡蛋")
    cfg = server.ServerConfig(engine_url=f"http://127.0.0.1:{engine.server_port}/v1")
    httpd = server.create_server("127.0.0.1", 0, cfg)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        text = _post_wav(httpd.server_port)
        assert text == "1. 买牛奶\n2. 买鸡蛋\n\n"
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
