import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

import pytest


SERVER_DIR = Path(__file__).resolve().parents[1]


def load_filter_module():
    spec = importlib.util.spec_from_file_location(
        "filler_filter", SERVER_DIR / "filler_filter.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


filter_fillers = load_filter_module().filter_fillers


def test_removes_isolated_fillers():
    assert filter_fillers("嗯，我觉得可以") == "我觉得可以"
    assert filter_fillers("我今天，呃，去了公司") == "我今天，去了公司"
    assert filter_fillers("三点了。嗯，那我先挂了。") == "三点了。那我先挂了。"
    assert filter_fillers("今天下雨，嗯，我带伞了") == "今天下雨，我带伞了"
    assert filter_fillers("嗯嗯，好的") == "好的"
    assert filter_fillers("哎呀，忘带钥匙了") == "忘带钥匙了"


def test_removes_glued_fillers_from_unpunctuated_speech():
    """ASR often emits no pauses, so fillers are glued to real words."""
    assert filter_fillers("啊那官方的呃飞猪云") == "那官方的飞猪云"
    assert filter_fillers("啊我们明天去") == "我们明天去"
    assert filter_fillers("呃我其实想说") == "我其实想说"
    assert filter_fillers("官方的呃飞猪云") == "官方的飞猪云"
    assert filter_fillers("哦对了我忘了") == "对了我忘了"


def test_stall_cluster_between_clauses_becomes_comma():
    assert filter_fillers("版本啊嗯而且我们要改") == "版本，而且我们要改"
    assert filter_fillers("采用了3.0啊嗯目前很成熟") == "采用了3.0，目前很成熟"
    assert filter_fillers("贡献了啊然后为什么") == "贡献了，然后为什么"


def test_keeps_meaningful_interjections_particles_and_demonstratives():
    assert filter_fillers("好的啊") == "好的啊"
    assert filter_fillers("我说啊，这个不行") == "我说啊，这个不行"
    assert filter_fillers("是这个吧") == "是这个吧"
    assert filter_fillers("就选那个。") == "就选那个。"
    assert filter_fillers("拿这个来") == "拿这个来"
    assert filter_fillers("哎哟好疼") == "哎哟好疼"
    assert filter_fillers("哎呀这怎么办") == "哎呀这怎么办"
    assert filter_fillers("方案就是说，得改") == "方案就是说，得改"
    assert filter_fillers("意思也就是说，你走吧") == "意思也就是说，你走吧"


def test_drops_trailing_demonstrative_echo():
    # "火啊这个" is a topical echo after a stall; "改一下这个" is a real object.
    assert filter_fillers("为什么这个项目会这么火啊这个") == "为什么这个项目会这么火啊"
    assert filter_fillers("这个我们改一下这个") == "这个我们改一下这个"


def test_removes_segment_initial_demonstrative_filler_only():
    assert filter_fillers("那个，文件我看过了") == "文件我看过了"
    assert filter_fillers("那个文件我看过了") == "那个文件我看过了"
    assert filter_fillers("这个这个，我们来商量") == "我们来商量"


def test_keeps_meaningful_connectors():
    assert filter_fillers("然后呢？然后我们再说") == "然后呢？然后我们再说"
    assert filter_fillers("好的，那就这样") == "好的，那就这样"


def test_removes_bilingual_stall_words():
    assert filter_fillers("um, I think we should go") == "I think we should go"
    assert filter_fillers("Hmm, let me check") == "let me check"
    assert filter_fillers("underline is a word") == "underline is a word"


def test_tidy_punctuation_after_removal():
    assert filter_fillers("你先走，嗯，我马上来") == "你先走，我马上来"
    assert filter_fillers("嗯，第一点买牛奶") == "第一点买牛奶"
    assert filter_fillers("好的。嗯。我们走吧。") == "好的。我们走吧。"


def test_preserves_line_boundaries():
    assert filter_fillers("嗯，第一点\n呃，第二点") == "第一点\n第二点"


def test_preserves_ascii_ellipsis_and_spacing():
    assert filter_fillers("um ... well") == "... well"
    assert filter_fillers("喂？喂？能听到吗") == "喂？喂？能听到吗"


@pytest.mark.parametrize(
    "text", ["", "   ", "正常文本", "没有填充词的完整句子。"],
)
def test_no_fillers_leaves_text_alone(text):
    assert filter_fillers(text) == text.strip()


# --- pipeline integration: filter_fillers is wired into the transcription path.
# The server module imports sibling modules; resolve them like production (run
# from the funasr-server dir). Scoped to this file so the other test suite keeps
# running against its own import context.


def load_server_module():
    # Server runs from the funasr-server dir in production, so sibling imports
    # (itn_digits / command_canon / filler_filter) resolve there. Resolve the
    # same way here, then undo the path/registry side effects so the other test
    # suite in this session keeps running against its own (identity) fallbacks.
    if str(SERVER_DIR) not in sys.path:
        sys.path.insert(0, str(SERVER_DIR))
    spec = importlib.util.spec_from_file_location(
        "funasr_gguf_server", SERVER_DIR / "funasr_gguf_server.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["funasr_gguf_server"] = module
    spec.loader.exec_module(module)
    for name in ("itn_digits", "command_canon", "filler_filter", "funasr_gguf_server"):
        sys.modules.pop(name, None)
    if sys.path and sys.path[0] == str(SERVER_DIR):
        sys.path.pop(0)
    return module


def _fake_binary(tmp_path, text):
    path = tmp_path / "fake_funasr.py"
    path.write_text("#!/usr/bin/env python3\nprint(%r)\n" % text, encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def _transcribe_json(httpd, server):
    boundary = "----funasr-filler-test-boundary"
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


def _start_server(module, cfg):
    httpd = module.create_server("127.0.0.1", 0, cfg)
    import threading

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def test_rule_organizer_strips_filler_before_numbering(tmp_path):
    """A leading filler must not hide a 第一点 enumeration marker."""
    server = load_server_module()
    cfg = server.ServerConfig(
        binary=_fake_binary(tmp_path, "嗯，第一点买牛奶\n呃，第二点买鸡蛋"),
        model=str(tmp_path / "m.gguf"),
        work_dir=str(tmp_path),
        organizer="rule",
    )
    httpd = _start_server(server, cfg)
    try:
        payload = _transcribe_json(httpd, server)
        assert payload["text"] == "1. 买牛奶\n2. 买鸡蛋\n\n"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_rule_organizer_removes_mid_sentence_filler(tmp_path):
    server = load_server_module()
    cfg = server.ServerConfig(
        binary=_fake_binary(tmp_path, "我今天，呃，去了公司，嗯，然后开会"),
        model=str(tmp_path / "m.gguf"),
        work_dir=str(tmp_path),
        organizer="rule",
    )
    httpd = _start_server(server, cfg)
    try:
        payload = _transcribe_json(httpd, server)
        assert payload["text"] == "我今天，去了公司，然后开会\n\n"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_none_organizer_keeps_filler_raw(tmp_path):
    server = load_server_module()
    cfg = server.ServerConfig(
        binary=_fake_binary(tmp_path, "嗯，马上来"),
        model=str(tmp_path / "m.gguf"),
        work_dir=str(tmp_path),
        organizer="none",
    )
    httpd = _start_server(server, cfg)
    try:
        payload = _transcribe_json(httpd, server)
        assert payload["text"] == "嗯，马上来\n\n"
    finally:
        httpd.shutdown()
        httpd.server_close()
