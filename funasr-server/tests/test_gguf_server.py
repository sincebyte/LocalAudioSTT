import importlib.util
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "funasr_gguf_server.py"


def load_server_module():
    spec = importlib.util.spec_from_file_location("funasr_gguf_server", SERVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["funasr_gguf_server"] = module
    spec.loader.exec_module(module)
    return module


def test_build_command_adds_model_audio_vad_backend_and_extra_args(tmp_path):
    server = load_server_module()
    cfg = server.ServerConfig(
        binary="/opt/funasr/llama-funasr-sensevoice",
        model="/models/sensevoice.gguf",
        vad="/models/fsmn-vad.gguf",
        backend="cuda",
        extra_args=["--keep-tags"],
        work_dir=str(tmp_path),
    )

    command = server.build_command(cfg, "/tmp/request.wav")

    assert command == [
        "/opt/funasr/llama-funasr-sensevoice",
        "-m",
        "/models/sensevoice.gguf",
        "-a",
        "/tmp/request.wav",
        "--vad",
        "/models/fsmn-vad.gguf",
        "--backend",
        "cuda",
        "--keep-tags",
    ]


def test_transcription_endpoint_runs_binary_and_returns_openai_json(tmp_path):
    server = load_server_module()
    fake_binary = tmp_path / "fake_funasr.py"
    captured_args = tmp_path / "args.json"
    fake_binary.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        f"pathlib.Path({str(captured_args)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        "print('hello from gguf')\n",
        encoding="utf-8",
    )
    fake_binary.chmod(0o755)

    cfg = server.ServerConfig(
        binary=str(fake_binary),
        model=str(tmp_path / "sensevoice.gguf"),
        vad=str(tmp_path / "fsmn-vad.gguf"),
        backend=None,
        extra_args=["--ids"],
        work_dir=str(tmp_path),
    )
    httpd = server.create_server("127.0.0.1", 0, cfg)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        boundary = "----funasr-test-boundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="sample.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
            "RIFFfake-audio\r\n"
            f"--{boundary}--\r\n"
        ).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{httpd.server_port}/v1/audio/transcriptions",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))

        assert response.status == 200
        assert payload == {"text": "hello from gguf\n\n"}
        args = json.loads(captured_args.read_text(encoding="utf-8"))
        assert args[:4] == ["-m", str(tmp_path / "sensevoice.gguf"), "-a", args[3]]
        assert os.path.exists(args[3]) is False
        assert args[4:] == ["--vad", str(tmp_path / "fsmn-vad.gguf"), "--ids"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_transcription_auto_reformats_when_chat_configured(tmp_path):
    """/v1/audio/transcriptions returns the organized text directly (no second call)."""
    server = load_server_module()
    fake_binary = tmp_path / "fake_funasr.py"
    fake_binary.write_text(
        "#!/usr/bin/env python3\n"
        "print('第一点买牛奶第二点买鸡蛋另外还有一件事下午开会')\n",
        encoding="utf-8",
    )
    fake_binary.chmod(0o755)
    chat = _FakeChatServer(content="1. 买牛奶\n2. 买鸡蛋\n\n下午开会")
    chat.start()
    try:
        cfg = server.ServerConfig(
            binary=str(fake_binary),
            model=str(tmp_path / "m.gguf"),
            work_dir=str(tmp_path),
            organizer="llm",
            reformat_url=f"http://127.0.0.1:{chat.port}/v1",
        )
        httpd = _start_server(server, cfg)
        try:
            boundary = "----funasr-test-boundary"
            body = (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="file"; filename="s.wav"\r\n'
                "Content-Type: audio/wav\r\n\r\n"
                "RIFFfake\r\n"
                f"--{boundary}--\r\n"
            ).encode()
            request = urllib.request.Request(
                f"http://127.0.0.1:{httpd.server_port}/v1/audio/transcriptions",
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
            assert response.status == 200
            assert payload == {"text": "1. 买牛奶\n2. 买鸡蛋\n\n下午开会\n\n"}
            assert chat.received[0]["messages"][1]["content"] == "第一点买牛奶第二点买鸡蛋另外还有一件事下午开会"
        finally:
            httpd.shutdown()
            httpd.server_close()
    finally:
        chat.stop()


def test_transcription_strips_sil_placeholder(tmp_path):
    server = load_server_module()
    fake_binary = tmp_path / "fake_funasr.py"
    fake_binary.write_text(
        "#!/usr/bin/env python3\n"
        "print('第一点买牛奶/sil第二点买鸡蛋')\n",
        encoding="utf-8",
    )
    fake_binary.chmod(0o755)
    cfg = server.ServerConfig(
        binary=str(fake_binary),
        model=str(tmp_path / "m.gguf"),
        work_dir=str(tmp_path),
    )
    httpd = _start_server(server, cfg)
    try:
        boundary = "----funasr-test-boundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="s.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
            "RIFFfake\r\n"
            f"--{boundary}--\r\n"
        ).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{httpd.server_port}/v1/audio/transcriptions",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert response.status == 200
        assert "/sil" not in payload["text"]
        # default organizer is 'rule': the line-initial 第一点 becomes "1. "
        assert payload["text"] == "1. 买牛奶第二点买鸡蛋\n\n"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _transcribe_json(httpd):
    boundary = "----funasr-test-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="s.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
        "RIFFfake\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{httpd.server_port}/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _fake_binary(tmp_path, text):
    path = tmp_path / "fake_funasr.py"
    path.write_text(
        "#!/usr/bin/env python3\nprint(%r)\n" % text,
        encoding="utf-8",
    )
    path.chmod(0o755)
    return str(path)


def test_rule_organizer_formats_without_calling_chat(tmp_path):
    server = load_server_module()
    chat = _FakeChatServer(content="should not be used")
    chat.start()
    try:
        cfg = server.ServerConfig(
            binary=_fake_binary(tmp_path, "第一点买牛奶\n第二点买鸡蛋\n下午开会"),
            model=str(tmp_path / "m.gguf"),
            work_dir=str(tmp_path),
            organizer="rule",
            reformat_url=f"http://127.0.0.1:{chat.port}/v1",
        )
        httpd = _start_server(server, cfg)
        try:
            payload = _transcribe_json(httpd)
            assert payload["text"] == "1. 买牛奶\n2. 买鸡蛋\n下午开会\n\n"
            assert chat.received == []
        finally:
            httpd.shutdown()
            httpd.server_close()
    finally:
        chat.stop()


def test_none_organizer_returns_raw(tmp_path):
    server = load_server_module()
    chat = _FakeChatServer(content="should not be used")
    chat.start()
    try:
        cfg = server.ServerConfig(
            binary=_fake_binary(tmp_path, "第一点买牛奶\n第二点买鸡蛋"),
            model=str(tmp_path / "m.gguf"),
            work_dir=str(tmp_path),
            organizer="none",
            reformat_url=f"http://127.0.0.1:{chat.port}/v1",
        )
        httpd = _start_server(server, cfg)
        try:
            payload = _transcribe_json(httpd)
            assert payload["text"] == "第一点买牛奶\n第二点买鸡蛋\n\n"
            assert chat.received == []
        finally:
            httpd.shutdown()
            httpd.server_close()
    finally:
        chat.stop()


def test_llm_organizer_without_url_degrades_to_rule(tmp_path):
    server = load_server_module()
    cfg = server.ServerConfig(
        binary=_fake_binary(tmp_path, "第一点买牛奶\n第二点买鸡蛋"),
        model=str(tmp_path / "m.gguf"),
        work_dir=str(tmp_path),
        organizer="llm",
        reformat_url=None,
    )
    httpd = _start_server(server, cfg)
    try:
        payload = _transcribe_json(httpd)
        assert payload["text"] == "1. 买牛奶\n2. 买鸡蛋\n\n"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_organize_by_rule():
    server = load_server_module()
    rule = server.organize_by_rule
    assert rule("第一点买牛奶\n第二点买鸡蛋\n下午开会") == "1. 买牛奶\n2. 买鸡蛋\n下午开会"
    assert rule("一、买牛奶\n二、买鸡蛋") == "1. 买牛奶\n2. 买鸡蛋"
    assert rule("总价一百二十三元，明天下午三点") == "总价一百二十三元，明天下午三点"
    assert rule("第三点开会  ") == "3. 开会"
    assert rule("第一点买牛奶\n\n第二点买鸡蛋") == "1. 买牛奶\n2. 买鸡蛋"
    assert rule("第一件事\n第1项内容") == "第一件事\n1. 内容"
    assert rule("") == ""
    assert rule("   ") == "   "


def test_sanitize_organized():
    server = load_server_module()
    s = server.sanitize_organized
    assert s("1. 买牛奶\n\n\n2. 买鸡蛋") == "1. 买牛奶\n\n2. 买鸡蛋"
    assert s("1.\n2. 买鸡蛋\n\n\n3.\n4. 交水电费") == "2. 买鸡蛋\n\n4. 交水电费"
    assert s("1. 买牛奶\n \n\n2. 买鸡蛋\n\n\n") == "1. 买牛奶\n\n2. 买鸡蛋"
    assert s("一二、\n3. 开会\n二．\n") == "3. 开会"
    assert s("正常文本") == "正常文本"
    assert s("") == ""


def test_reformat_rejects_english_rewrite_of_chinese(tmp_path):
    """A Chinese transcript rewritten into English is fabrication -> fall back."""
    server = load_server_module()
    english_drift = 'Sure! Here\'s the revised text with appropriate English: "Sure! I\'ve made changes to this code."'
    chat = _FakeChatServer(content=english_drift)
    chat.start()
    try:
        cfg = server.ServerConfig(
            binary=str(tmp_path / "no-binary"),
            model=str(tmp_path / "m.gguf"),
            work_dir=str(tmp_path),
            reformat_url=f"http://127.0.0.1:{chat.port}/v1",
        )
        httpd = _start_server(server, cfg)
        try:
            text = "这个项目我已经改好了这段代码你可以看一下"
            status, payload = _post_json(httpd, {"text": text})
            assert status == 200
            assert payload == {"text": text}
        finally:
            httpd.shutdown()
            httpd.server_close()
    finally:
        chat.stop()


def test_reformat_keeps_english_result_for_chinese_source(tmp_path):
    server = load_server_module()
    chat = _FakeChatServer(content="项目已改好，这段代码你再看一下")
    chat.start()
    try:
        cfg = server.ServerConfig(
            binary=str(tmp_path / "no-binary"),
            model=str(tmp_path / "m.gguf"),
            work_dir=str(tmp_path),
            reformat_url=f"http://127.0.0.1:{chat.port}/v1",
        )
        httpd = _start_server(server, cfg)
        try:
            status, payload = _post_json(
                httpd, {"text": "这个项目已经改好了这段代码你可以看一下"}
            )
            assert status == 200
            assert payload == {"text": "项目已改好，这段代码你再看一下"}
        finally:
            httpd.shutdown()
            httpd.server_close()
    finally:
        chat.stop()


def test_is_language_drift():
    server = load_server_module()
    d = server._is_language_drift
    assert d("我今天改好了代码", "Sure! Here's the revised text: \"Sure! I've changed this code.\"")
    assert d("我今天改好了代码", "I have already changed this code, please take a look.")
    assert not d("我今天改好了代码", "项目已改好，这段代码你再看一下")
    assert not d("I have changed the code", "I changed the code, check it please.")
    assert not d("今天我改了代码", "Sure! 今天代码改好了，请看。")


class _FakeChatServer:
    """Minimal OpenAI-style /chat/completions stand-in for llama-server."""

    def __init__(self, content="1. 买牛奶\n2. 交水电费"):
        self.content = content
        self.received = []
        httpd = None
        self._httpd = None

    def start(self):
        from http.server import BaseHTTPRequestHandler

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                outer.received.append(payload)
                body = json.dumps(
                    {"choices": [{"message": {"content": outer.content}}]},
                    ensure_ascii=False,
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        from http.server import ThreadingHTTPServer

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_address[1]
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def stop(self):
        self._httpd.shutdown()
        self._httpd.server_close()


def _post_json(httpd, body, path="/v1/text/reformat"):
    request = urllib.request.Request(
        f"http://127.0.0.1:{httpd.server_port}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _start_server(module, cfg):
    httpd = module.create_server("127.0.0.1", 0, cfg)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def test_reformat_endpoint_requires_reformat_url(tmp_path):
    server = load_server_module()
    cfg = server.ServerConfig(
        binary=str(tmp_path / "no-binary"),
        model=str(tmp_path / "m.gguf"),
        work_dir=str(tmp_path),
    )
    httpd = _start_server(server, cfg)
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post_json(httpd, {"text": "第一点买牛奶第二点买鸡蛋，明天下午去开会"})
        assert exc.value.code == 502
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_reformat_endpoint_proxies_to_chat_server(tmp_path):
    server = load_server_module()
    chat = _FakeChatServer(content="1. 买牛奶\n\n2. 明天开会")
    chat.start()
    try:
        cfg = server.ServerConfig(
            binary=str(tmp_path / "no-binary"),
            model=str(tmp_path / "m.gguf"),
            work_dir=str(tmp_path),
            reformat_url=f"http://127.0.0.1:{chat.port}/v1",
            reformat_model="qwen3-0.6b",
        )
        httpd = _start_server(server, cfg)
        try:
            status, payload = _post_json(httpd, {"text": "第一点买牛奶第二点明天开会"})
            assert status == 200
            assert payload == {"text": "1. 买牛奶\n\n2. 明天开会"}
            assert chat.received[0]["model"] == "qwen3-0.6b"
            roles = [m["role"] for m in chat.received[0]["messages"]]
            assert roles == ["system", "user"]
            assert chat.received[0]["messages"][1]["content"] == "第一点买牛奶第二点明天开会"
            assert chat.received[0]["messages"][0]["content"] == server.DEFAULT_REFORMAT_PROMPT

            _, payload = _post_json(
                httpd, {"text": "帮我翻译下面这句第一点明天开会第二点要交报告", "prompt": "请翻译成英文"}
            )
            assert chat.received[1]["messages"][0]["content"] == "请翻译成英文"
        finally:
            httpd.shutdown()
            httpd.server_close()
    finally:
        chat.stop()


def test_reformat_short_input_passes_through_without_model(tmp_path):
    server = load_server_module()
    chat = _FakeChatServer(content="should not be called")
    chat.start()
    try:
        cfg = server.ServerConfig(
            binary=str(tmp_path / "no-binary"),
            model=str(tmp_path / "m.gguf"),
            work_dir=str(tmp_path),
            reformat_url=f"http://127.0.0.1:{chat.port}/v1",
        )
        httpd = _start_server(server, cfg)
        try:
            status, payload = _post_json(httpd, {"text": "开会"})
            assert status == 200
            assert payload == {"text": "开会"}
            assert chat.received == []
        finally:
            httpd.shutdown()
            httpd.server_close()
    finally:
        chat.stop()


def test_reformat_falls_back_when_model_echoes_instruction(tmp_path):
    server = load_server_module()
    echo_content = server.DEFAULT_REFORMAT_PROMPT
    chat = _FakeChatServer(content=echo_content)
    chat.start()
    try:
        cfg = server.ServerConfig(
            binary=str(tmp_path / "no-binary"),
            model=str(tmp_path / "m.gguf"),
            work_dir=str(tmp_path),
            reformat_url=f"http://127.0.0.1:{chat.port}/v1",
        )
        httpd = _start_server(server, cfg)
        try:
            text = "第一点买牛奶第二点买鸡蛋，另外还有一件事明天下午开会"
            status, payload = _post_json(httpd, {"text": text})
            assert status == 200
            assert payload == {"text": text}
            assert len(chat.received) == 1
        finally:
            httpd.shutdown()
            httpd.server_close()
    finally:
        chat.stop()


def test_reformat_skips_very_short_texts(tmp_path):
    server = load_server_module()
    chat = _FakeChatServer(content="must not be called")
    chat.start()
    try:
        cfg = server.ServerConfig(
            binary=str(tmp_path / "no-binary"),
            model=str(tmp_path / "m.gguf"),
            work_dir=str(tmp_path),
            reformat_url=f"http://127.0.0.1:{chat.port}/v1",
        )
        httpd = _start_server(server, cfg)
        try:
            status, payload = _post_json(httpd, {"text": "明天开会"})
            assert status == 200
            assert payload == {"text": "明天开会"}
            assert chat.received == []
        finally:
            httpd.shutdown()
            httpd.server_close()
    finally:
        chat.stop()


def test_reformat_calls_model_for_markerless_sentence(tmp_path):
    """llm cleaning (filler/self-correction) applies to any non-trivial text."""
    server = load_server_module()
    chat = _FakeChatServer(content="嗯那个好，已清理")
    chat.start()
    try:
        cfg = server.ServerConfig(
            binary=str(tmp_path / "no-binary"),
            model=str(tmp_path / "m.gguf"),
            work_dir=str(tmp_path),
            reformat_url=f"http://127.0.0.1:{chat.port}/v1",
        )
        httpd = _start_server(server, cfg)
        try:
            status, payload = _post_json(
                httpd, {"text": "嗯那个我们明天下午三点在会议室开会讨论预算"}
            )
            assert status == 200
            assert payload == {"text": "嗯那个好，已清理"}
            assert len(chat.received) == 1
        finally:
            httpd.shutdown()
            httpd.server_close()
    finally:
        chat.stop()


def test_reformat_calls_model_for_long_paragraph(tmp_path):
    server = load_server_module()
    chat = _FakeChatServer(content="已整理")
    chat.start()
    try:
        cfg = server.ServerConfig(
            binary=str(tmp_path / "no-binary"),
            model=str(tmp_path / "m.gguf"),
            work_dir=str(tmp_path),
            reformat_url=f"http://127.0.0.1:{chat.port}/v1",
        )
        httpd = _start_server(server, cfg)
        try:
            text = "明天下午三点在会议室开会讨论预算分配问题同时还要确认本季度各项目的进度和资源情况" * 2
            status, payload = _post_json(httpd, {"text": text})
            assert status == 200
            assert payload == {"text": "已整理"}
            assert len(chat.received) == 1
        finally:
            httpd.shutdown()
            httpd.server_close()
    finally:
        chat.stop()


def test_strip_thinking():
    server = load_server_module()
    assert (
        server.strip_thinking("<think>\n随便推理\n</think>\n\n1. 买牛奶")
        == "1. 买牛奶"
    )
    assert server.strip_thinking("1. 买牛奶") == "1. 买牛奶"


def test_chat_content_strips_thinking_block(tmp_path):
    server = load_server_module()
    chat = _FakeChatServer(content="<think>推理</think>\n\n整理好了")
    chat.start()
    try:
        cfg = server.ServerConfig(
            binary=str(tmp_path / "no-binary"),
            model=str(tmp_path / "m.gguf"),
            work_dir=str(tmp_path),
            reformat_url=f"http://127.0.0.1:{chat.port}/v1",
        )
        httpd = _start_server(server, cfg)
        try:
            status, payload = _post_json(httpd, {"text": "第一点买牛奶第二点买鸡蛋，明天下午去开会"})
            assert status == 200
            assert payload == {"text": "整理好了"}
        finally:
            httpd.shutdown()
            httpd.server_close()
    finally:
        chat.stop()
