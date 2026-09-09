#!/usr/bin/env python3
"""OpenAI-compatible HTTP wrapper around llama.cpp's Qwen3-ASR-1.7B server.

The heavy lifting is done by a long-lived llama.cpp `llama-server` (see
start-server.sh) that loads Qwen3-ASR-1.7B Q8_0 + mmproj and exposes its own
OpenAI-compatible `POST /v1/audio/transcriptions` on an internal port. This
wrapper proxies that endpoint and runs the deterministic text-cleanup pipeline
(spoken-filler removal, enumeration formatting, empty-transcript gating) on top,
so callers get clean, paragraph-separated text in one request.

  POST /v1/audio/transcriptions   (multipart `file` wav; `model` optional)
  GET  /health

There is deliberately NO alias/dictionary rewriting and NO Chinese->Arabic
numeral conversion here - Qwen3-ASR-1.7B handles mixed zh/en well enough that
those mappings are unnecessary.
"""

from __future__ import annotations

import argparse
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

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/health":
            self._send_json(HTTPStatus.OK, _json_bytes({"status": "ok"}))
            return
        self._send_json(HTTPStatus.NOT_FOUND, _error_payload("not found"))

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/v1/audio/transcriptions":
            self._send_json(HTTPStatus.NOT_FOUND, _error_payload("not found"))
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            content_type = self.headers.get("Content-Type", "")
            audio_bytes, filename = parse_multipart_file(content_type, self.rfile.read(length))
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

        text = finalize_transcription(raw, organizer=self.config.organizer)
        self._send_json(HTTPStatus.OK, _json_bytes({"text": text}))


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
        timeout=args.timeout,
    )
    httpd = create_server(args.host, args.port, config)
    print(f"Serving Qwen3-ASR transcription on http://{args.host}:{httpd.server_port}", flush=True)
    print(f"engine: {config.engine_url}  organizer: {config.organizer}  prompt: {'set' if config.prompt else 'default'}  temp: {config.temperature}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
