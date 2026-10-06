"""Logging setup: plain text for dev, JSON lines for containers, with a per-session correlation id."""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar

session_id_var: ContextVar[str] = ContextVar("session_id", default="-")


class _SessionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = session_id_var.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "session_id": getattr(record, "session_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", fmt: str = "text") -> None:
    handler = logging.StreamHandler()
    handler.addFilter(_SessionFilter())
    handler.setFormatter(
        _JsonFormatter()
        if fmt == "json"
        else logging.Formatter("%(asctime)s %(levelname)-7s [%(session_id)s] %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # Third-party chatter that is noise at INFO.
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("google_adk.google.adk.telemetry._metrics").setLevel(logging.ERROR)
