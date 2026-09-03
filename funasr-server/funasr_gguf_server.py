#!/usr/bin/env python3
"""OpenAI-compatible HTTP wrapper for FunASR llama.cpp / GGUF binaries.

This server intentionally keeps inference in the existing C++ command-line
tools. It accepts a multipart audio upload, runs the configured GGUF binary, and
returns the transcript as a small OpenAI-compatible JSON response.

By default each request spawns a fresh CLI subprocess (one-shot; simplest, but
reloads ~1GB of models every time). With --persistent the server keeps a single
long-lived `llama-funasr-cli --server` worker alive so the models stay resident
in memory and per-request latency drops to roughly the pure inference time.
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
import subprocess
import tempfile
import threading
from typing import Iterable, Optional
from urllib.parse import urlparse


@dataclass(frozen=True)
class ServerConfig:
    binary: str
    model: str
    vad: Optional[str] = None
    backend: Optional[str] = None
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
    command.extend(config.extra_args)
    return command


def build_worker_command(config: ServerConfig) -> list[str]:
    """Command line for the long-lived --server worker (no per-file -a arg)."""
    command = [config.binary, "-m", config.model, "--server"]
    if config.vad:
        command.extend(["--vad", config.vad])
    if config.backend:
        command.extend(["--backend", config.backend])
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

        self._send_json(HTTPStatus.OK, _json_bytes({"text": transcript}))


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
        extra_args=_parse_extra_args(args.extra_arg),
        work_dir=args.work_dir,
        timeout=args.timeout,
        transcriber=transcriber,
    )
    httpd = create_server(args.host, args.port, config)
    print(f"Serving FunASR GGUF transcription on http://{args.host}:{httpd.server_port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
