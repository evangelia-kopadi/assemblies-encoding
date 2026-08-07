"""Logging helpers for readable experiment output."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TextIO

LOGGER_NAMESPACE = "assemblies_encoding"


def _coerce_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    level_name = level.upper()
    if not hasattr(logging, level_name):
        raise ValueError(f"Unknown logging level: {level!r}")
    return int(getattr(logging, level_name))


def configure_logging(
    *,
    level: int | str | None = logging.INFO,
    log_file: str | Path | None = None,
    stream: TextIO | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the repository logger.

    The console handler intentionally uses message-only formatting so existing
    experiment output stays readable while callers can lower/raise verbosity or
    add file logging for long sweeps.
    """
    logger = logging.getLogger(LOGGER_NAMESPACE)
    if level is not None:
        logger.setLevel(_coerce_level(level))
    logger.propagate = False

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    if not any(getattr(handler, "_assemblies_console", False) for handler in logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout if stream is None else stream)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        console_handler._assemblies_console = True  # type: ignore[attr-defined]
        logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_log_path = str(log_path.resolve())
        if not any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename).resolve() == Path(resolved_log_path)
            for handler in logger.handlers
        ):
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            )
            logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the repository logging namespace."""
    logger = logging.getLogger(LOGGER_NAMESPACE)
    if not logger.handlers:
        configure_logging()
    module_name = name[4:] if name.startswith("src.") else name
    return logging.getLogger(f"{LOGGER_NAMESPACE}.{module_name}")
