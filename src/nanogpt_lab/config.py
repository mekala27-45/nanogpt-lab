"""Typed configuration for NanoGPT-Lab, loaded from a YAML file."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DataCfg(BaseModel):
    corpus: str
    url: str
    sha256: str
    cache_dir: str
    val_fraction: float


class ModelCfg(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    dropout: float
    mlp_ratio: int
    rope_theta: float
    tie_weights: bool
    norm: Literal["rms", "layer"]
    pos: Literal["rope", "learned"]
    mlp: Literal["swiglu", "gelu"]

    @property
    def head_dim(self) -> int:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        return self.n_embd // self.n_head


class TrainCfg(BaseModel):
    batch_size: int
    max_iters: int
    eval_interval: int
    eval_iters: int
    warmup_iters: int
    lr: float
    min_lr: float
    weight_decay: float
    grad_clip: float
    device: str
    compile: bool
    out_dir: str


class GenCfg(BaseModel):
    max_new_tokens: int
    temperature: float
    top_k: int
    prompt: str


class Config(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    seed: int
    data: DataCfg
    model: ModelCfg
    train: TrainCfg
    generate: GenCfg


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", populate_by_name=True, protected_namespaces=()
    )
    log_level: str = "INFO"
    log_format: str = "console"


def load_config(path: Path | str) -> Config:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Config.model_validate(raw)


def resolve_device(preference: str) -> str:
    """Map 'auto'/'cuda'/'cpu' to an available device string."""
    if preference == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_settings() -> Settings:
    return Settings()
