from __future__ import annotations

import random as _python_random
import time
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch

from . import hardware, random


def _slice_batch(batch: Any, start: int, stop: int) -> Any:
    if torch.is_tensor(batch) or isinstance(batch, np.ndarray):
        return batch[start:stop]
    if isinstance(batch, tuple):
        return tuple(_slice_batch(item, start, stop) for item in batch)
    if isinstance(batch, list):
        return [_slice_batch(item, start, stop) for item in batch]
    if isinstance(batch, dict):
        return {key: _slice_batch(value, start, stop) for key, value in batch.items()}
    return batch[start:stop]


def _batch_length(batch: Any) -> int:
    if isinstance(batch, dict):
        first_key = next(iter(batch))
        return len(batch[first_key])
    return len(batch)


def iter_batches(batch: Any, batch_size: int) -> Iterator[Any]:
    total = _batch_length(batch)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        yield _slice_batch(batch, start, stop)


def improve_reproducibility(seed: int) -> None:
    _python_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class Timer:
    _start_time: float | None = None

    def run(self) -> "Timer":
        self._start_time = time.time()
        return self

    def reset(self) -> "Timer":
        return self.run()

    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def __str__(self) -> str:
        elapsed = self.elapsed()
        minutes, seconds = divmod(int(elapsed), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


__all__ = [
    "Timer",
    "hardware",
    "improve_reproducibility",
    "iter_batches",
    "random",
]
