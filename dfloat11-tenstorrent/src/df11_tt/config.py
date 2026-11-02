"""Central configuration for df11-tt project."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    model_name: str = "Llama-3.2-1B-Instruct"
    hf_model_path: str = ""
    tt_metal_home: str = str(Path.home() / "tt-metal")
    device_id: int = 0
    num_layers: Optional[int] = None
    max_seq_len: int = 1024
    batch_size: int = 1
    instruct: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EvalConfig:
    prompt: str = "What is the capital of France?"
    max_generated_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 0.08
    top_k: int = 32
    warmup_runs: int = 0
    num_runs: int = 1
    verify_correctness: bool = True
    verbose: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "EvalConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CompressionConfig:
    codec: str = "huffman_exponent"
    layer_targets: list = field(default_factory=lambda: ["lm_head"])
    artifact_dir: str = "outputs/artifacts"
    verify: bool = True
    verbose: bool = True
