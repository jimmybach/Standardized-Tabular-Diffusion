from __future__ import annotations

import torch


def get_gpus_info() -> list[dict[str, object]]:
    if not torch.cuda.is_available():
        return []

    gpus: list[dict[str, object]] = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        gpus.append(
            {
                "index": idx,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "multi_processor_count": props.multi_processor_count,
            }
        )
    return gpus
