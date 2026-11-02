"""Simple context-manager timer."""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TimingRecord:
    durations: Dict[str, float] = field(default_factory=dict)

    def record(self, name: str, duration_s: float):
        self.durations[name] = duration_s

    def get_ms(self, name: str) -> float:
        return self.durations.get(name, 0.0) * 1000.0

    def summary(self) -> Dict[str, float]:
        return {k: round(v * 1000, 3) for k, v in self.durations.items()}


class Timer:
    def __init__(self):
        self._start = None
        self.elapsed_s = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_s = time.perf_counter() - self._start

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_s * 1000.0
