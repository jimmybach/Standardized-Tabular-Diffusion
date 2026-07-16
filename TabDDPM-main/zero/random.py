from __future__ import annotations

import random as _python_random
from typing import Any

import numpy as np
import torch


def get_state() -> dict[str, Any]:
    return {
        "python": _python_random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_state(state: dict[str, Any]) -> None:
    _python_random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
