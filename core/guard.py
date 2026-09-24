"""Failure isolation: a crashing module degrades the game instead of killing it.

The assignment requires "module failures handled in a controlled way". Every module call in
the main loop goes through `ModuleGuard.call`, which returns a fallback value on an exception.
The first traceback per module is logged; after that failures are only counted and shown in
the debug overlay.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Callable

log = logging.getLogger(__name__)


class ModuleGuard:
    def __init__(self):
        self.failures: Counter[str] = Counter()
        self.last_error: dict[str, str] = {}

    def call(self, name: str, fn: Callable, *args, default: Any = None, **kwargs) -> Any:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, this is the safety net
            if self.failures[name] == 0:
                log.exception("Module '%s' failed; using fallback from now on when it fails", name)
            self.failures[name] += 1
            self.last_error[name] = f"{type(exc).__name__}: {exc}"
            return default
