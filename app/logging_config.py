"""
Central logging configuration.

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
log_config.py.

WHY LEVEL COMES FROM CONFIG:
Development wants DEBUG (see everything). Production wants INFO (skip the
noise). Driving this from settings.app_env means no code change between
environments — just the APP_ENV value in .env.

THIRD-PARTY NOISE:
transformers, sentence-transformers, httpx and friends emit very chatty
INFO/DEBUG logs. We raise their threshold to WARNING so our own logs
aren't drowned out.
"""

import logging

from app.config import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

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


def setup_logging() -> None:
    """
    Configure the root logger once. Call this at application startup,
    before any real work runs.
    """
    level = logging.DEBUG if settings.APP_ENV != "production" else logging.INFO

    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
    )

    # Quiet the noisy third-party libraries.
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured (env=%s, level=%s)",
        settings.APP_ENV,
        logging.getLevelName(level),
    )