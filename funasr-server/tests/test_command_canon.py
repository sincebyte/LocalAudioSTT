import importlib.util
import json
import os
from pathlib import Path

import pytest


SERVER_DIR = Path(__file__).resolve().parents[1]


def load_canon_module():
    spec = importlib.util.spec_from_file_location(
        "command_canon", SERVER_DIR / "command_canon.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


canon = load_canon_module()
normalize = canon.normalize_commands


@pytest.fixture()
def alias_file(tmp_path, monkeypatch):
    """Point FUNASR_COMMAND_ALIASES_FILE at a per-test file (empty by default)
    so tests never touch the shipped command_aliases.json."""
    path = tmp_path / "aliases.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("FUNASR_COMMAND_ALIASES_FILE", str(path))
    return path


def _write(payload: dict, path: Path) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_default_fuzzy_clear_matches_nearby_english():
    assert normalize("claer the screen") == "clear the screen"
    assert normalize("please clear the screen") == "please clear the screen"


def test_default_send_aliases_rewrite_observed_spelling():
    assert normalize("请发松一下") == "请发送一下"
    assert normalize("发宋文件") == "发送文件"


def test_spelled_out_letters_join_before_matching():
    assert normalize("C L E A R") == "clear"
    assert normalize("please C L E A R now") == "please clear now"
    assert normalize("c l e a r it") == "clear it"


def test_file_profile_rewrites_ascii_and_cjk_forms(alias_file):
    _write({"org model": ["og model", "奥格猫抖", "奥格莫得"]}, alias_file)
    assert normalize("og model ready") == "org model ready"
    assert normalize("奥格猫抖有新版") == "org model 有新版"


def test_cjk_alias_to_ascii_keeps_reading_spacing(alias_file):
    _write({"emacs": ["伊马克斯", "emax"], "funasr": ["fun asr", "芬阿斯尔"]}, alias_file)
    assert normalize("伊马克斯很强大") == "emacs 很强大"
    assert normalize("用 emax 编辑") == "用 emacs 编辑"
    assert normalize("芬阿斯尔在本地运行") == "funasr 在本地运行"


def test_file_replaces_default_key_wholesale(alias_file):
    _write({"clear": ["克丽尔"]}, alias_file)
    assert normalize("克丽尔") == "clear"
    assert normalize("可丽儿") == "可丽儿"  # default list replaced -> no match
    assert normalize("claer") == "clear"  # fuzzy stays on unless disabled


def test_dict_form_disables_fuzzy(alias_file):
    _write({"clear": {"forms": ["克丽尔"], "fuzzy": False}}, alias_file)
    assert normalize("克丽尔") == "clear"
    assert normalize("claer") == "claer"  # fuzzy off -> untouched
    assert normalize("可丽儿") == "可丽儿"


def test_env_aliases_append_on_top(alias_file, monkeypatch):
    _write({"org model": ["og model"]}, alias_file)
    monkeypatch.setenv(
        "FUNASR_COMMAND_ALIASES",
        json.dumps({"org model": ["奥格猫抖"]}, ensure_ascii=False),
    )
    assert normalize("og model") == "org model"
    assert normalize("奥格猫抖") == "org model"


def test_matching_is_case_insensitive(alias_file):
    _write({"asr": ["a s r"], "org model": ["OG model", "o r g model"]}, alias_file)
    assert normalize("ASR 很好用") == "asr 很好用"
    assert normalize("A S R 很好用") == "asr 很好用"  # spelled letters joined
    assert normalize("OG MODEL 已就绪") == "org model 已就绪"


def test_no_configured_match_leaves_text_alone():
    assert normalize("好好") == "好好"
    assert normalize("我们明天下午三点开会") == "我们明天下午三点开会"
    assert normalize("this is a normal english sentence") == "this is a normal english sentence"
    assert normalize("I am ok") == "I am ok"
    assert normalize("") == ""
