"""Metrics data classes for baseline and compressed runs."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict


@dataclass
class MetricsReport:
    model_name: str = ""
    mode: str = ""  # "baseline" or "compressed"
    layer_name: str = ""
    original_bytes: int = 0
    compressed_bytes: int = 0
    compression_ratio: float = 0.0
    bytes_saved: int = 0
    decode_ms: float = 0.0
    upload_ms: float = 0.0
    tt_op_ms: float = 0.0
    total_layer_ms: float = 0.0
    baseline_e2e_ms: float = 0.0
    compressed_e2e_ms: float = 0.0
    tokens_per_sec: Optional[float] = None
    ttft_ms: Optional[float] = None
    bit_exact: Optional[bool] = None
    token_match: Optional[bool] = None
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ComparisonReport:
    baseline: MetricsReport = field(default_factory=MetricsReport)
    compressed: MetricsReport = field(default_factory=MetricsReport)
    overhead_ms: float = 0.0
    overhead_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "baseline": self.baseline.to_dict(),
            "compressed": self.compressed.to_dict(),
            "overhead_ms": self.overhead_ms,
            "overhead_pct": self.overhead_pct,
        }
