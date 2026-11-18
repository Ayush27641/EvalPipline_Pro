"""Minimal structured-ish logging setup (stdlib only, Python 3.9 compatible).

One line per record, UTC timestamps, logger name included so RAG / eval / judge
logs are filterable in `docker compose logs api`.
"""

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if root.handlers:  # uvicorn --reload re-imports the app; don't stack handlers
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
    root.addHandler(handler)
    root.setLevel(level)

    # Third-party chatter we never want at INFO
    for noisy in ("httpx", "httpcore", "sentence_transformers", "transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
