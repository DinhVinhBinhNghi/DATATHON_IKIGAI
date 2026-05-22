"""Logging utilities: get_logger() trả về logger format thống nhất,
timed() context manager đo elapsed time + log.

Pattern:
    logger = get_logger(__name__)
    with timed("build covisit matrix", logger):
        # ... heavy work ...
    # logs: "  done build covisit matrix: 12.3s"
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

# Config logger 1 lần (idempotent)
_CONFIGURED = False


def _configure_root() -> None:
    """Configure root logger lần đầu, đọc format từ config."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    # Import inside để tránh circular import (config dùng nothing from logging)
    try:
        from src.common.config import get_config
        cfg = get_config()
        level = getattr(cfg.logging, "level", "INFO")
        fmt = getattr(cfg.logging, "fmt",
                      "[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        datefmt = getattr(cfg.logging, "datefmt", "%H:%M:%S")
    except Exception:
        # Fallback nếu config chưa setup
        level = "INFO"
        fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
        datefmt = "%H:%M:%S"

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root = logging.getLogger()
    # Chỉ add handler nếu chưa có (tránh duplicate logs)
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Trả về logger với name. Tự configure root nếu chưa.

    Recommended: gọi get_logger(__name__) ở đầu mỗi module.
    """
    _configure_root()
    return logging.getLogger(name)


@contextmanager
def timed(label: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Context manager log thời gian thực thi.

    Args:
        label: mô tả tác vụ.
        logger: logger để emit message. Nếu None, dùng print().

    Example:
        with timed("preaggregate 500 files", logger):
            do_heavy_work()
        # → logs: "  done preaggregate 500 files: 1234.5s"
    """
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        msg = f"  done {label}: {dt:.1f}s"
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)
