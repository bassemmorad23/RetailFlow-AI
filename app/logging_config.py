"""
Central logging configuration.

WHY JSON LOGS:
Every log line is a JSON object with consistent fields. This makes logs
machine-readable — filterable by store_id/conversation_id, aggregatable
for metrics, and directly consumable by tools like Sentry, Datadog, and
CloudWatch. Plain-text logs are readable by humans but useless for
querying at scale.

WHY THIS IS CONFIGURED ONCE, CENTRALLY:
Every module already creates its own logger via
`logging.getLogger(__name__)` and calls logger.info/warning/error. But
until something calls logging.basicConfig (or equivalent), those calls
use Python's bare default and are inconsistent or invisible. This file
configures the ROOT logger once at startup so every module's logs share
one format and destination, with zero per-module setup.

WHY NOT NAMED logging.py:
A module named logging.py would shadow Python's stdlib `logging` and
break `import logging` across the whole codebase — the same trap that
made us rename the analytics folder away from `logging/`. Hence
logging_config.py.

WHY LEVEL COMES FROM CONFIG:
Development wants DEBUG (see everything). Production wants INFO (skip the
noise). Driving this from settings.APP_ENV means no code change between
environments — just the APP_ENV value in .env.

THIRD-PARTY NOISE:
transformers, sentence-transformers, httpx and friends emit very chatty
INFO/DEBUG logs. We raise their threshold to WARNING so our own logs
aren't drowned out.

UVICORN LOGGERS:
Uvicorn ships with its own loggers (uvicorn, uvicorn.error,
uvicorn.access) that install their own plain-text handlers on startup.
We clear those handlers and let them propagate to the root, so uvicorn's
own logs share the same JSON format as ours — otherwise the terminal is
a mix of two formats.

ADDING CONTEXT FIELDS TO A LOG LINE:
Any call can attach structured fields using the `extra` argument:

    logger.info("received message", extra={
        "store_id": message.store_id,
        "conversation_id": message.conversation_id,
        "request_id": request_id,
    })

Those keys become top-level keys in the resulting JSON. Standard fields
(timestamp, level, logger name, message) are always included.
"""

import logging
from xml.sax import handler

from pythonjsonlogger.json import JsonFormatter

from app.config import settings
from app.log_context import get_request_context
from logging.handlers import RotatingFileHandler
from pathlib import Path



_LOG_FILE_PATH = Path("data/logs/app.jsonl")
_LOG_FILE_MAX_BYTES = 10 * 1024 * 1024   # 10 MB per file
_LOG_FILE_BACKUP_COUNT = 5                # keeps 5 rotated backups → 50MB max total


# Third-party loggers that are too verbose at INFO/DEBUG.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "transformers",
    "sentence_transformers",
    "openai",
    "pymongo",
)


class _ContextFilter(logging.Filter):
    """
    Injects per-request context (request_id, store_id, conversation_id)
    into every log record. Fields set via log_context.set_request_context()
    appear automatically in the JSON output; fields that weren't set stay
    as None (visible in logs but obviously empty).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in get_request_context().items():
            setattr(record, key, value)
        return True


def setup_logging() -> None:
    """
    Configure the root logger once. Call this at application startup,
    before any real work runs.
    """
    level = logging.DEBUG if settings.APP_ENV != "production" else logging.INFO

    # Every log record becomes a JSON object with these standard fields,
    # plus any extra fields the call site provides via extra={...}.
    formatter = JsonFormatter(
        "{asctime} {levelname} {name} {message} {request_id} {store_id} {conversation_id}",
        style="{",
        rename_fields={
            "asctime": "timestamp",
            "levelname": "level",
            "name": "logger",
        },
    )
    
    # Terminal handler — for live watching during dev/debug

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(_ContextFilter())

    # File handler — rotating JSONL for later analysis and cross-restart persistence
    _LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        _LOG_FILE_PATH,
        maxBytes=_LOG_FILE_MAX_BYTES,
        backupCount=_LOG_FILE_BACKUP_COUNT,
        encoding="utf-8",
    )
    
    file_handler.setFormatter(formatter)
    file_handler.addFilter(_ContextFilter())

    root = logging.getLogger()
    root.setLevel(level)

    # Both handlers active — every log line goes to terminal AND file
    root.handlers = [stream_handler, file_handler]

    # Quiet the noisy third-party libraries.
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Force uvicorn's own loggers to propagate through our root handler,
    # so uvicorn access/error logs are ALSO JSON-formatted.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True

    logging.getLogger(__name__).info(
        "Logging configured",
        extra={
            "env": settings.APP_ENV,
            "level": logging.getLevelName(level),
        },
    )