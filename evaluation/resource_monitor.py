"""Low-overhead, opt-in process resource monitoring for simulation or Pi runs."""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Union

try:  # Keep evaluation imports usable on development machines without psutil.
    import psutil
except ImportError:  # pragma: no cover - depends on deployment environment
    psutil = None

PathLike = Union[str, Path]


@dataclass(frozen=True)
class ResourceSample:
    timestamp_s: float
    cpu_percent: float
    memory_bytes: int
    network_sent_bytes: int
    network_received_bytes: int


class ResourceMonitor:
    """Capture process CPU/RAM plus host network counters at explicit intervals."""

    def __init__(self, pid: Optional[int] = None) -> None:
        if psutil is None:
            raise RuntimeError("ResourceMonitor requires psutil; install the project dependencies first")
        self._process = psutil.Process(pid)
        self._process.cpu_percent(None)  # Prime psutil's delta-based CPU measurement.
        self.samples: List[ResourceSample] = []

    def sample(self) -> ResourceSample:
        network = psutil.net_io_counters()
        value = ResourceSample(
            timestamp_s=time.time(),
            cpu_percent=self._process.cpu_percent(None),
            memory_bytes=self._process.memory_info().rss,
            network_sent_bytes=network.bytes_sent,
            network_received_bytes=network.bytes_recv,
        )
        self.samples.append(value)
        return value

    def write_csv(self, path: PathLike) -> None:
        """Persist the collected samples without creating parent directories implicitly."""
        destination = Path(path)
        if not destination.parent.exists():
            raise FileNotFoundError(f"output directory does not exist: {destination.parent}")
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ResourceSample.__dataclass_fields__.keys())
            writer.writeheader()
            writer.writerows(asdict(sample) for sample in self.samples)
