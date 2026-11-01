#!/usr/bin/env python3
"""OpenAI-compatible HTTP wrapper around llama.cpp's Qwen3-ASR-1.7B server.

The heavy lifting is done by a long-lived llama.cpp `llama-server` (see
start-server.sh) that loads Qwen3-ASR-1.7B bf16 + mmproj and exposes its own
OpenAI-compatible `POST /v1/audio/transcriptions` on an internal port. This
wrapper proxies that endpoint and runs the deterministic text-cleanup pipeline
(spoken-filler removal, numeral/command normalization, trailing-punctuation
drop, empty-transcript gating) on top, so callers get clean text in one request
with line breaks left as-is.

  POST /v1/audio/transcriptions   (multipart `file` wav; `model` optional)
  POST /v1/audio/transcriptions/stream  (same request; SSE deltas + final text)
  GET  /health

There is deliberately NO alias/dictionary rewriting and NO Chinese->Arabic
numeral conversion here - Qwen3-ASR-1.7B handles mixed zh/en well enough that
those mappings are unnecessary.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import request as urllib_request
from urllib.parse import urlparse
from urllib.error import HTTPError

from cleanup import finalize_transcription

ORGANIZER_VALUES = ("none", "rule")


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8011
    engine_url: str = "http://127.0.0.1:8083/v1"
    engine_model: str = "qwen3-asr-1.7b"
    organizer: str = "rule"
    prompt: str = ""   # empty -> llama.cpp's built-in ASR prompt is used
    temperature: float = 0.0  # deterministic greedy decoding for transcription
    strip_trailing_punct: bool = True  # drop ASR end-of-segment punctuation
    log_raw: bool = False  # log engine raw text + cleaned text (diagnostics)
    timeout: float = 120.0


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _error_payload(message: str) -> bytes:
    return _json_bytes({"error": {"message": message}})


def transcribe_via_engine(config: ServerConfig, audio_bytes: bytes, filename: str) -> str:
    """Forward one audio clip to the resident Qwen3-ASR llama-server."""
    url = config.engine_url.rstrip("/") + "/audio/transcriptions"
    boundary = "----qwen3asr-proxy-boundary"
    body = b""
    body += f"--{boundary}\r\n".encode()
    body += (
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode()
    body += audio_bytes
    body += b"\r\n"
    body += f"--{boundary}\r\n".encode()
    body += (
        'Content-Disposition: form-data; name="model"\r\n\r\n'
        f"{config.engine_model}\r\n"
    ).encode()
    if config.prompt:
        body += f"--{boundary}\r\n".encode()
        body += 'Content-Disposition: form-data; name="prompt"\r\n\r\n'.encode()
        body += config.prompt.encode("utf-8")
        body += b"\r\n"
    # Greedy decoding: llama.cpp's default temperature is 0.8, which makes
    # transcription nondeterministic. ASR should be stable, so force 0.0.
    body += f"--{boundary}\r\n".encode()
    body += 'Content-Disposition: form-data; name="temperature"\r\n\r\n'.encode()
    body += f"{config.temperature}\r\n".encode()
    body += f"--{boundary}--\r\n".encode()

    request = urllib_request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=config.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"engine {url} -> HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"engine {url} failed: {exc}") from exc
    try:
        text = payload["text"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"unexpected engine payload: {payload!r}") from exc
    if not isinstance(text, str):
        raise RuntimeError("engine returned non-string text")
    return text


ASR_TEXT_TAG = "<asr_text>"
# How long the marker filter buffers before assuming the engine emitted no
# language marker at all. "language Chinese<asr_text>" is far shorter than this.
_MARKER_HOLD_LIMIT = 64


class MarkerStreamFilter:
    """Incrementally drop Qwen3-ASR's `language ...<asr_text>` prefix.

    The engine streams that prefix token by token (`language`, ` Chinese`,
    `<asr_text>`), so a per-token strip is not enough: hold output until the tag
    appears, then emit only what follows it. If no tag shows up within a short
    prefix, pass everything through unchanged.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._done = False

    def feed(self, delta: str) -> str:
        if self._done:
            return delta
        self._buffer += delta
        index = self._buffer.lower().find(ASR_TEXT_TAG)
        if index >= 0:
            out = self._buffer[index + len(ASR_TEXT_TAG):]
            self._buffer = ""
            self._done = True
            return out
        if len(self._buffer) >= _MARKER_HOLD_LIMIT:
            out = self._buffer
            self._buffer = ""
            self._done = True
            return out
        return ""

    def flush(self) -> str:
        if self._done:
            return ""
        out = self._buffer
        self._buffer = ""
        self._done = True
        return out


def stream_raw_via_engine(config: ServerConfig, audio_bytes: bytes):
    """Yield raw transcript tokens for one clip from llama-server's SSE.

    `/v1/audio/transcriptions` returns one JSON body; `/v1/chat/completions`
    streams the same ASR decode token by token (`stream: true`), which is what
    the streaming endpoint forwards. The configured prompt travels as a text
    part so hotword/context bias matches the non-streaming endpoint.
    """
    url = config.engine_url.rstrip("/") + "/chat/completions"
    encoded = base64.b64encode(audio_bytes).decode("ascii")
    content = []
    if config.prompt:
        content.append({"type": "text", "text": config.prompt})
    content.append({
        "type": "input_audio",
        "input_audio": {"data": encoded, "format": "wav"},
    })
    payload = {
        "model": config.engine_model,
        "stream": True,
        "temperature": config.temperature,
        "messages": [{"role": "user", "content": content}],
    }
    request = urllib_request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=config.timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                    delta = event["choices"][0].get("delta", {}).get("content")
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue
                if delta:
                    yield delta
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"engine {url} -> HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"engine {url} failed: {exc}") from exc


def _is_file_field(part) -> bool:
    disposition = part.get("Content-Disposition", "")
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


class Qwen3ASRHandler(BaseHTTPRequestHandler):
    server_version = "Qwen3ASRServer/0.1"

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

    def _send_sse_headers(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        # Close-delimited body: no Content-Length, the stream ends with the connection.
        self.close_connection = True

    def _write_sse(self, payload: dict) -> None:
        line = "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
        self.wfile.write(line.encode("utf-8"))
        self.wfile.flush()

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/health":
            self._send_json(HTTPStatus.OK, _json_bytes({"status": "ok"}))
            return
        self._send_json(HTTPStatus.NOT_FOUND, _error_payload("not found"))

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/v1/audio/transcriptions":
            self._handle_transcription()
            return
        if path == "/v1/audio/transcriptions/stream":
            self._handle_stream_transcription()
            return
        self._send_json(HTTPStatus.NOT_FOUND, _error_payload("not found"))

    def _read_audio_request(self) -> tuple[bytes, str]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("missing Content-Length")
        content_type = self.headers.get("Content-Type", "")
        return parse_multipart_file(content_type, self.rfile.read(length))

    def _handle_transcription(self) -> None:
        try:
            audio_bytes, filename = self._read_audio_request()
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, _error_payload(str(exc)))
            return

        try:
            raw = transcribe_via_engine(self.config, audio_bytes, filename)
        except RuntimeError as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, _error_payload(str(exc)))
            return
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, _error_payload(str(exc)))
            return

        # Line breaks are the engine's own; the pipeline never adds or merges
        # them, so paragraph layout is left to the caller / a downstream LLM.
        text = finalize_transcription(
            raw,
            organizer=self.config.organizer,
            prompt=self.config.prompt,
            strip_trailing_punct=self.config.strip_trailing_punct,
        )
        if self.config.log_raw:
            # Raw vs cleaned, so a suspected duplicate can be traced to the
            # engine (both show it) or to the caller (raw is clean). repr keeps
            # newlines/whitespace visible. Diagnostics only.
            print(f"[raw]   {filename}: {raw!r}", flush=True)
            print(f"[clean] {filename}: {text!r}", flush=True)
        self._send_json(HTTPStatus.OK, _json_bytes({"text": text}))

    def _handle_stream_transcription(self) -> None:
        """Per-chunk streaming: SSE deltas as the engine decodes, then `final`.

        The caller sends one short audio chunk per request and keeps its own
        order; the deltas are for live display, while `final` is the cleaned
        text that should replace them (filler removal, numeral/command mapping,
        trailing-punctuation policy).
        """
        try:
            audio_bytes, filename = self._read_audio_request()
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, _error_payload(str(exc)))
            return

        self._send_sse_headers()
        raw_parts: list[str] = []
        marker_filter = MarkerStreamFilter()
        try:
            for delta in stream_raw_via_engine(self.config, audio_bytes):
                raw_parts.append(delta)
                visible = marker_filter.feed(delta)
                if visible:
                    self._write_sse({"type": "delta", "text": visible})
            tail = marker_filter.flush()
            if tail:
                self._write_sse({"type": "delta", "text": tail})
            raw = "".join(raw_parts)
            text = finalize_transcription(
                raw,
                organizer=self.config.organizer,
                prompt=self.config.prompt,
                strip_trailing_punct=self.config.strip_trailing_punct,
            )
            if self.config.log_raw:
                print(f"[stream raw]   {filename}: {raw!r}", flush=True)
                print(f"[stream clean] {filename}: {text!r}", flush=True)
            self._write_sse({"type": "final", "text": text})
            self._write_sse({"type": "done"})
        except (BrokenPipeError, ConnectionResetError):
            # Client cancelled (stop/clear): nothing left to report.
            return
        except Exception as exc:
            try:
                self._write_sse({"type": "error", "message": str(exc)})
            except Exception:
                pass


class Qwen3ASRHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_class, config: ServerConfig):
        super().__init__(server_address, handler_class)
        self.config = config


def create_server(host: str, port: int, config: ServerConfig) -> Qwen3ASRHTTPServer:
    return Qwen3ASRHTTPServer((host, port), Qwen3ASRHandler, config)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve cleaned Qwen3-ASR-1.7B transcription over OpenAI-compatible HTTP."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument(
        "--engine-url",
        default="http://127.0.0.1:8083/v1",
        help="Base URL of the resident llama.cpp Qwen3-ASR llama-server.",
    )
    parser.add_argument(
        "--engine-model", default="qwen3-asr-1.7b",
        help="Model id forwarded to the engine (not validated by llama.cpp).",
    )
    parser.add_argument(
        "--organizer",
        choices=ORGANIZER_VALUES,
        default="rule",
        help="none (raw transcript) or rule (filler cleanup + enumeration; default).",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional transcription prompt forwarded to the engine as hotword/"
        "context bias (OpenAI transcriptions 'prompt' field). Empty uses the "
        "engine's built-in ASR prompt. Default set by start-server.sh.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature sent to the engine. 0.0 = deterministic "
        "greedy (default; llama.cpp's own default 0.8 makes ASR vary).",
    )
    parser.add_argument(
        "--strip-trailing-punct",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop the sentence-ender the ASR adds at the very end of each "
        "pause-flushed segment, so the downstream LLM decides the final "
        "punctuation (default on; --no-strip-trailing-punct keeps it).",
    )
    parser.add_argument(
        "--log-raw",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log each request's engine raw text and cleaned text to stderr "
        "(goes to server.log). Diagnostics for suspected ASR duplicates; off "
        "by default.",
    )
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Engine request timeout in seconds.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = ServerConfig(
        host=args.host,
        port=args.port,
        engine_url=args.engine_url,
        engine_model=args.engine_model,
        organizer=args.organizer,
        prompt=args.prompt,
        temperature=args.temperature,
        strip_trailing_punct=args.strip_trailing_punct,
        log_raw=args.log_raw,
        timeout=args.timeout,
    )
    httpd = create_server(args.host, args.port, config)
    print(f"Serving Qwen3-ASR transcription on http://{args.host}:{httpd.server_port}", flush=True)
    print(f"engine: {config.engine_url}  organizer: {config.organizer}  prompt: {'set' if config.prompt else 'default'}  temp: {config.temperature}  strip_trailing_punct: {config.strip_trailing_punct}  log_raw: {config.log_raw}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
