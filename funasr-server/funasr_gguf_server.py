#!/usr/bin/env python3
"""OpenAI-compatible HTTP wrapper for FunASR llama.cpp / GGUF binaries.

This server intentionally keeps inference in the existing C++ command-line
tools. It accepts a multipart audio upload, runs the configured GGUF binary, and
returns the transcript as a small OpenAI-compatible JSON response.

By default each request spawns a fresh CLI subprocess (one-shot; simplest, but
reloads ~1GB of models every time). With --persistent the server keeps a single
long-lived `llama-funasr-cli --server` worker alive so the models stay resident
in memory and per-request latency drops to roughly the pure inference time.

POST /v1/audio/transcriptions already returns the *organized* text in one shot:
the raw transcript is auto-run through a whole-buffer reorganization pass
(Feishu-style numbering / blank lines / numeral normalization) and a trailing
blank line is appended, so consecutive voice inserts read as separate
paragraphs. The reorganization needs a *general* chat model, not the
Fun-ASR-tuned one used for audio, so it is forwarded to a separate llama.cpp
llama-server (--reformat-url) hosting e.g. a stock Qwen3-0.6B GGUF.

POST /v1/text/reformat stays available for manually reorganizing a larger
accumulated buffer in one call.
"""

from __future__ import annotations

import argparse
import atexit
import base64
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import tempfile
import threading
from typing import Iterable, Optional
from urllib import request as urllib_request
from urllib.parse import urlparse
from urllib.error import HTTPError

try:
    from itn_digits import normalize_chinese_digits
except ImportError:  # pragma: no cover - module co-located in this package
    def normalize_chinese_digits(text: str) -> str:  # type: ignore[misc]
        return text

try:
    from command_canon import normalize_commands
except ImportError:  # pragma: no cover - module co-located in this package
    def normalize_commands(text: str) -> str:  # type: ignore[misc]
        return text


# Default system instruction for /v1/text/reformat. The Fun-ASR-tuned Qwen3
# used for transcription cannot follow pure-text instructions (it stops
# immediately), so this runs on the general chat model at --reformat-url.
#
# Goal: turn raw speech-to-text into clean, correct, readable text - drop
# self-corrections and filler words, and do NOT force any numbering.
# It is copy-editing, not summarization: every sentence must stay, verbatim
# in meaning and order.
DEFAULT_REFORMAT_PROMPT = (
    "你是语音转写的校对整理助手，只做逐句校对排版、不做摘要，按原文保留每一句话和全部内容：\n"
    "1) 改口误：说错又纠正的只留最后正确的说法，删“不对”“说错了”等修正语；\n"
    "2) 删口头语：删“嗯、啊、呃、那个、就是说”等没意义的填充词；\n"
    "3) 数字一律转阿拉伯数字：三十→30、百分之二十→20%、三点半→3点半、三十五块→35块、"
    "二〇二五→2025、第一点→第1点，单位量词保留；固定成语除外（如“不三不四”）；\n"
    "4) 补恰当中文标点；不要加“好的”等开场白；不要复述本指令或解释；\n"
    "语言与原文一致：中文原文输出中文、英文原文输出英文。"
)

# Distinctive phrases that only ever come from the instruction itself. If two or
# more appear in a reply, the model echoed the rules instead of organizing input.
_INSTRUCTION_MARKERS = (
    "语音转写的校对整理助手",
    "只做逐句校对排版",
    "按原文保留每一句话",
    "改口误",
    "删口头语",
    "数字一律转阿拉伯数字",
    "单位量词保留",
    "固定成语除外",
    "不要复述本指令",
    "中文原文输出中文",
)

# Qwen3 thinking models emit a <think>...</think> block before the answer.
_THINK_RE = re.compile(r"^<think>.*?</think>\s*", re.DOTALL)

# Inputs shorter than this are not sent to the llm organizer (too little to
# clean; the 0.6B model would otherwise just parrot the instruction back).
REFORMAT_MIN_LENGTH = 12


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


# Organizer modes for the text returned by /v1/audio/transcriptions:
#   none - raw transcript only (no reorganization at all)
#   rule - deterministic local formatting, no second model call
#   llm  - full reorganization through the general chat model
ORGANIZER_VALUES = ("none", "rule", "llm")

# Line-initial Chinese ordinals that organize_by_rule turns into numbered lines.
_CN_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_HEAD_CN_ORDINAL_RE = re.compile(r"^第\s*([一二三四五六七八九十]|[0-9]+)\s*[点条项部分]")
_HEAD_CN_PUNCT_RE = re.compile(r"^([一二三四五六七八九十])\s*[、.．]")


def _to_arabic(num: str) -> str:
    if num.isdigit():
        return str(int(num))
    if num in _CN_DIGITS:
        return str(_CN_DIGITS[num])
    return num


def organize_by_rule(text: str) -> str:
    """Deterministic, model-free formatting of one transcription request.

    Only rewrites clean line-initial enumeration markers into numbered lines
    (第X点/一、... -> "N. ..."); everything else is left untouched. This cannot
    split unpunctuated run-on lists - that is the known trade-off for skipping
    the language model.
    """
    if not text.strip():
        return text
    lines_out = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        match = _HEAD_CN_ORDINAL_RE.match(line)
        if match:
            head = _to_arabic(match.group(1))
            rest = line[match.end():]
            line = f"{head}. {rest}".rstrip()
        else:
            match = _HEAD_CN_PUNCT_RE.match(line)
            if match:
                head = _to_arabic(match.group(1))
                rest = line[match.end():]
                line = f"{head}. {rest}".rstrip()
        lines_out.append(line)
    return "\n".join(lines_out)


def _effective_organizer(config: "ServerConfig") -> str:
    """Resolve the organizer to run for transcription.

    'llm' needs a configured chat backend; without one it degrades to 'rule' so
    transcription never depends on the optional reformat server.
    """
    if config.organizer == "llm" and not config.reformat_url:
        return "rule"
    return config.organizer


def _looks_like_instruction_echo(result: str, system: str) -> bool:
    """True when the model parroted the instruction instead of organizing input."""
    compact = _norm(result)
    if len(compact) < 8:
        return False
    # Exact lead of the instruction copied verbatim.
    probe = _norm(system)[:16]
    if bool(probe) and probe in compact:
        return True
    # The model often drops the opening line and re-emits the numbered rules
    # instead, so match on several instruction-only phrases.
    return sum(1 for marker in _INSTRUCTION_MARKERS if _norm(marker) in compact) >= 2


def strip_thinking(text: str) -> str:
    """Remove a leading Qwen3-style <think>...</think> reasoning block."""
    return _THINK_RE.sub("", text, count=1)


@dataclass(frozen=True)
class ServerConfig:
    binary: str
    model: str
    vad: Optional[str] = None
    backend: Optional[str] = None
    prompt: Optional[str] = None
    organizer: str = "rule"
    reformat_url: Optional[str] = None
    reformat_model: str = "qwen3-0.6b"
    reformat_prompt: Optional[str] = None
    reformat_timeout: float = 300.0
    extra_args: list[str] = field(default_factory=list)
    work_dir: str = field(default_factory=tempfile.gettempdir)
    timeout: float = 600.0
    transcriber: Optional[PersistentTranscriber] = None


def build_command(config: ServerConfig, audio_path: str) -> list[str]:
    command = [config.binary, "-m", config.model, "-a", audio_path]
    if config.vad:
        command.extend(["--vad", config.vad])
    if config.backend:
        command.extend(["--backend", config.backend])
    if config.prompt:
        command.extend(["--prompt", config.prompt])
    command.extend(config.extra_args)
    return command


def build_worker_command(config: ServerConfig) -> list[str]:
    """Command line for the long-lived --server worker (no per-file -a arg)."""
    command = [config.binary, "-m", config.model, "--server"]
    if config.vad:
        command.extend(["--vad", config.vad])
    if config.backend:
        command.extend(["--backend", config.backend])
    if config.prompt:
        command.extend(["--prompt", config.prompt])
    command.extend(config.extra_args)
    return command


def extract_transcript(stdout: str) -> str:
    return stdout.strip()


def transcribe_file(config: ServerConfig, audio_path: str) -> str:
    completed = subprocess.run(
        build_command(config, audio_path),
        check=True,
        capture_output=True,
        text=True,
        timeout=config.timeout,
    )
    return extract_transcript(completed.stdout)


def _chat_completion(config: ServerConfig, system: str, user: str) -> str:
    """Call the configured llama-server (OpenAI /v1/chat/completions)."""
    if not config.reformat_url:
        raise RuntimeError(
            "no --reformat-url configured for /v1/text/reformat; start the "
            "reformat llama-server (see start-funasr-server.sh / README)"
        )
    url = config.reformat_url.rstrip("/") + "/chat/completions"
    # The Qwen3 chat model may emit a <think> block before the answer, and with
    # too small a cap the reasoning eats the whole budget and we get an empty
    # reply. Give a generous budget (input-length scaled), then strip <think>
    # afterwards; the echo-detection fallback still guards against runaway loops.
    max_tokens = max(512, min(2048, int(len(user) * 2.0) + 256))
    body = json.dumps(
        {
            "model": config.reformat_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib_request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib_request.urlopen(request, timeout=config.reformat_timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"reformat upstream {url} -> HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"reformat upstream {url} failed: {exc}") from exc
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected reformat upstream payload: {payload!r}") from exc
    if not isinstance(content, str):
        raise RuntimeError("reformat upstream returned non-string content")
    return content


# A numbered line with nothing after the number (empty bullet the model can emit).
_EMPTY_NUMBER_LINE_RE = re.compile(r"^[0-9一二三四五六七八九十]+\s*[.、．]\s*$")


def sanitize_organized(text: str) -> str:
    """Clean reorganization output: drop empty numbered lines and collapse
    repeated blank lines, so a run of blank bullets never reaches the caller."""
    lines = text.split("\n")
    out: list[str] = []
    prev_blank = False
    for raw in lines:
        line = raw.strip()
        if not line:
            if out and not prev_blank:
                out.append("")
                prev_blank = True
            continue
        prev_blank = False
        if _EMPTY_NUMBER_LINE_RE.match(line):
            continue
        out.append(line)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")


def _is_language_drift(source: str, result: str) -> bool:
    """True when a mostly-Chinese transcript came back rewritten in English.

    The general chat model sometimes 'improves' short/noisy text into a fluent
    English rewrite ("Sure! Here's the revised text..."). That is fabrication,
    not organizing, so it must fall back to the original transcript.
    """
    src_cjk = len(_CJK_RE.findall(source))
    src_en = len(_ASCII_LETTER_RE.findall(source))
    src_sig = src_cjk + src_en
    if src_sig < 6 or src_cjk < src_en:
        return False  # source is English/balanced - no Chinese to protect
    out_cjk = len(_CJK_RE.findall(result))
    out_en = len(_ASCII_LETTER_RE.findall(result))
    out_sig = out_cjk + out_en
    if out_sig < 6:
        return False
    return out_cjk == 0 or out_en >= out_sig * 0.6


def reformat_text_via_chat(config: ServerConfig, text: str, prompt: Optional[str]) -> str:
    """Whole-buffer reorganization through the general chat model.

    Returns text with a leading Qwen3 <think> block stripped, if present.
    Inputs without listable structure, instruction echoes and language-drift
    (Chinese rewritten into English) all fall back to the original.
    """
    if len(_norm(text)) < REFORMAT_MIN_LENGTH:
        return text
    system = prompt or config.reformat_prompt or DEFAULT_REFORMAT_PROMPT
    result = sanitize_organized(strip_thinking(_chat_completion(config, system, text)))
    if not result.strip() or _looks_like_instruction_echo(result, system):
        return text
    if _is_language_drift(text, result):
        print("[reformat] language drift detected (Chinese -> English rewrite), returned original",
              file=sys.stderr, flush=True)
        return text
    return result


def _strip_sil(text: str) -> str:
    """Drop the '/sil' placeholder the decoder emits for silence/noise segments."""
    return text.replace("/sil", "")


class PersistentTranscriber:
    """Keeps one `llama-funasr-cli --server` subprocess alive so the models stay
    resident in memory across requests.

    Protocol (one framed line per request on the worker's stdout):
      READY                         -> models loaded, worker accepting work
      B64 <base64(transcript)>      -> success
      ERR <base64(error message)>   -> failure

    The caller writes one absolute audio path + '\\n' to stdin per request.
    Requests are serialized with a lock (inference saturates the CPU anyway).
    """

    def __init__(self, config: ServerConfig):
        self._config = config
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen[str]] = None
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._start_worker()

    def _start_worker(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc.wait()
        proc = subprocess.Popen(
            build_worker_command(self._config),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit -> worker logs land in the server log
            text=True,
            bufsize=1,
        )
        self._proc = proc

        ready = proc.stdout.readline()
        if not ready.strip() or not ready.startswith("READY"):
            raise RuntimeError(
                f"persistent worker did not become ready: {ready!r} "
                f"(check the server log for model load errors)"
            )

        self._queue = queue.Queue()
        reader = threading.Thread(
            target=self._read_stdout,
            name="funasr-worker-reader",
            daemon=True,
        )
        reader.start()

    def _read_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        while True:
            line = proc.stdout.readline()
            if not line:
                self._queue.put(None)
                return
            line = line.strip()
            if line.startswith(("B64 ", "ERR ")):
                self._queue.put(line)

    def transcribe(self, audio_path: str) -> str:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("persistent worker not running")
        with self._lock:
            self._proc.stdin.write(audio_path + "\n")
            self._proc.stdin.flush()
            try:
                line = self._queue.get(timeout=self._config.timeout)
            except queue.Empty:
                self._reset_worker()
                raise subprocess.TimeoutExpired(
                    str(self._proc.args), self._config.timeout
                )
            if line is None:
                self._reset_worker()
                raise RuntimeError("persistent worker exited unexpectedly")
            tag, payload = line.split(" ", 1)
            payload = payload.strip()
            if tag == "B64":
                return base64.b64decode(payload).decode("utf-8").rstrip("\n")
            if tag == "ERR":
                message = base64.b64decode(payload).decode("utf-8")
                raise RuntimeError(message)
            raise RuntimeError(f"unexpected worker frame: {line!r}")

    def _reset_worker(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        try:
            self._start_worker()
        except Exception:
            self._proc = None

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _error_payload(message: str) -> bytes:
    return _json_bytes({"error": {"message": message}})


def _field_content_disposition(part) -> str:
    return part.get("Content-Disposition", "")


def _is_file_field(part) -> bool:
    disposition = _field_content_disposition(part)
    return "form-data" in disposition and 'name="file"' in disposition


def parse_multipart_file(content_type: str, body: bytes) -> tuple[bytes, str]:
    message = BytesParser(policy=policy.default).parsebytes(
        (
            f"Content-Type: {content_type}\r\n"
            "MIME-Version: 1.0\r\n"
            "\r\n"
        ).encode("utf-8")
        + body
    )
    if not message.is_multipart():
        raise ValueError("expected multipart/form-data")

    for part in message.iter_parts():
        if not _is_file_field(part):
            continue
        filename = part.get_filename() or "audio.wav"
        payload = part.get_payload(decode=True) or b""
        if not payload:
            raise ValueError("uploaded file is empty")
        return payload, filename
    raise ValueError("missing multipart field: file")


def _suffix_from_filename(filename: str) -> str:
    suffix = Path(filename).suffix
    if not suffix or len(suffix) > 16:
        return ".wav"
    return suffix


class FunASRGGUFHandler(BaseHTTPRequestHandler):
    server_version = "FunASRGGUFServer/0.1"

    def log_message(self, fmt: str, *args) -> None:
        print("%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), fmt % args))

    @property
    def config(self) -> ServerConfig:
        return self.server.config  # type: ignore[attr-defined]

    def _send_json(self, status: HTTPStatus, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/health":
            self._send_json(HTTPStatus.OK, _json_bytes({"status": "ok"}))
            return
        self._send_json(HTTPStatus.NOT_FOUND, _error_payload("not found"))

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/v1/text/reformat":
            self._handle_reformat()
            return
        if path != "/v1/audio/transcriptions":
            self._send_json(HTTPStatus.NOT_FOUND, _error_payload("not found"))
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            content_type = self.headers.get("Content-Type", "")
            audio_bytes, filename = parse_multipart_file(content_type, self.rfile.read(length))
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, _error_payload(str(exc)))
            return

        tmp_path = None
        try:
            os.makedirs(self.config.work_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "wb",
                suffix=_suffix_from_filename(filename),
                dir=self.config.work_dir,
                delete=False,
            ) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            if self.config.transcriber is not None:
                transcript = self.config.transcriber.transcribe(tmp_path)
            else:
                transcript = transcribe_file(self.config, tmp_path)
        except subprocess.CalledProcessError as exc:
            message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, _error_payload(message))
            return
        except subprocess.TimeoutExpired:
            self._send_json(HTTPStatus.GATEWAY_TIMEOUT, _error_payload("transcription timed out"))
            return
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, _error_payload(str(exc)))
            return
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except FileNotFoundError:
                    pass

        self._send_json(HTTPStatus.OK, _json_bytes({"text": self._finalize_transcription(transcript)}))

    def _finalize_transcription(self, transcript: str) -> str:
        """Raw ASR text -> text returned by /v1/audio/transcriptions.

        Applies the configured organizer so callers get the reorganized version
        in a single request:
          none - raw transcript
          rule - deterministic local formatting (no second model call)
          llm  - full reorganization through the general chat model
        A trailing blank line is always appended so consecutive voice inserts
        read as separate paragraphs.
        """
        text = _strip_sil(transcript)
        organizer = _effective_organizer(self.config)
        if organizer == "llm":
            try:
                organized = reformat_text_via_chat(self.config, transcript, None)
                if organized.strip():
                    text = organized
            except Exception:
                pass  # never let a formatting failure break transcription
        elif organizer == "rule":
            text = organize_by_rule(text)
        text = normalize_chinese_digits(text)
        text = normalize_commands(text)
        if not text.strip():
            return ""
        return text.rstrip("\n") + "\n\n"

    def _handle_reformat(self) -> None:
        """POST /v1/text/reformat — whole-buffer reorganization.

        Request:  {"text": "<accumulated raw transcript>", "prompt": "<optional>"}
        Response: {"text": "<reorganized transcript>"}

        Runs on the general chat model behind --reformat-url (the Fun-ASR-tuned
        audio model stops immediately on pure text). The caller replaces its
        whole buffer with the returned text.
        """
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("missing non-empty string field: text")
            prompt = payload.get("prompt")
            if prompt is not None and not isinstance(prompt, str):
                raise ValueError("field 'prompt' must be a string")
            result = reformat_text_via_chat(self.config, text, prompt)
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, _error_payload(str(exc)))
            return
        except RuntimeError as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, _error_payload(str(exc)))
            return
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, _error_payload(str(exc)))
            return
        if not result.strip():
            result = text  # never degrade: fall back to the original
        self._send_json(HTTPStatus.OK, _json_bytes({"text": result}))


class FunASRHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_class, config: ServerConfig):
        super().__init__(server_address, handler_class)
        self.config = config


def create_server(host: str, port: int, config: ServerConfig) -> FunASRHTTPServer:
    return FunASRHTTPServer((host, port), FunASRGGUFHandler, config)


def _parse_extra_args(values: Optional[Iterable[str]]) -> list[str]:
    args: list[str] = []
    for value in values or []:
        args.extend(value.split())
    return args


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a FunASR llama.cpp / GGUF binary over HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--binary", required=True, help="Path to llama-funasr-sensevoice or llama-funasr-paraformer.")
    parser.add_argument("--model", required=True, help="Path to the model GGUF passed as -m.")
    parser.add_argument("--vad", help="Optional FSMN-VAD GGUF passed as --vad.")
    parser.add_argument("--backend", choices=["cpu", "cuda"], help="Optional backend passed as --backend.")
    parser.add_argument(
        "--prompt",
        help="Optional text passed to the GGUF binary as --prompt: replaces the "
        "default user instruction ('语音转写：') with a custom hint, e.g. a command "
        "word list, to bias recognition. Applied to every request.",
    )
    parser.add_argument(
        "--organizer",
        choices=ORGANIZER_VALUES,
        default="rule",
        help="How the text returned by /v1/audio/transcriptions is organized: "
        "'none' (raw), 'rule' (local deterministic formatting, no extra model "
        "call; default), or 'llm' (full reorganization via --reformat-url). "
        "'llm' without a --reformat-url falls back to 'rule'.",
    )
    parser.add_argument(
        "--reformat-url",
        help="Base URL of a llama.cpp llama-server (OpenAI /v1 style, e.g. "
        "http://127.0.0.1:8082/v1) hosting a general chat model. Enables "
        "POST /v1/text/reformat (the Fun-ASR-tuned audio model cannot follow "
        "pure-text instructions).",
    )
    parser.add_argument(
        "--reformat-model",
        default="qwen3-0.6b",
        help="Model id sent in the chat-completion request to --reformat-url.",
    )
    parser.add_argument(
        "--reformat-prompt",
        help="Optional system instruction overriding the built-in default used by "
        "POST /v1/text/reformat (unless a request supplies its own 'prompt').",
    )
    parser.add_argument("--reformat-timeout", type=float, default=300.0)
    parser.add_argument("--work-dir", default=tempfile.gettempdir(), help="Directory for temporary uploaded audio files.")
    parser.add_argument("--timeout", type=float, default=600.0, help="Per-request subprocess timeout in seconds.")
    parser.add_argument(
        "--persistent",
        action="store_true",
        help="Keep one llama-funasr-cli --server worker alive so the models stay "
        "resident in memory between requests (removes the ~1s model-load cost). "
        "Requires a binary with --server support.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional argument(s) forwarded to the GGUF binary. May be repeated.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    transcriber = None
    if args.persistent:
        config = ServerConfig(
            binary=args.binary,
            model=args.model,
            vad=args.vad,
            backend=args.backend,
            prompt=args.prompt,
            organizer=args.organizer,
            reformat_url=args.reformat_url,
            reformat_model=args.reformat_model,
            reformat_prompt=args.reformat_prompt,
            reformat_timeout=args.reformat_timeout,
            extra_args=_parse_extra_args(args.extra_arg),
            work_dir=args.work_dir,
            timeout=args.timeout,
        )
        transcriber = PersistentTranscriber(config)
        atexit.register(transcriber.close)
        print("persistent worker started: models loaded into memory")
    config = ServerConfig(
        binary=args.binary,
        model=args.model,
        vad=args.vad,
        backend=args.backend,
        prompt=args.prompt,
        organizer=args.organizer,
        reformat_url=args.reformat_url,
        reformat_model=args.reformat_model,
        reformat_prompt=args.reformat_prompt,
        reformat_timeout=args.reformat_timeout,
        extra_args=_parse_extra_args(args.extra_arg),
        work_dir=args.work_dir,
        timeout=args.timeout,
        transcriber=transcriber,
    )
    httpd = create_server(args.host, args.port, config)
    print(f"Serving FunASR GGUF transcription on http://{args.host}:{httpd.server_port}", flush=True)
    print(f"organizer: {config.organizer}", flush=True)
    if config.reformat_url:
        print("==== reformat-prompt-begin ====", flush=True)
        print(config.reformat_prompt or DEFAULT_REFORMAT_PROMPT, flush=True)
        print("==== reformat-prompt-end ====", flush=True)
        print(
            f"Text reformat (/v1/text/reformat) proxied to {config.reformat_url} "
            f"(model {config.reformat_model})",
            flush=True,
        )
    httpd.serve_forever()


if __name__ == "__main__":
    main()
