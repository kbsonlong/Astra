"""避免训练 API 与应用组装代码之间的模型循环依赖。"""
from pydantic import BaseModel, Field
from typing import Literal

from .config import Qwen3TrainingConfig


class TrainingConfigPayload(BaseModel):
    model_path: str = Field(min_length=1, max_length=500)
    train_file: str = Field(min_length=1, max_length=1000)
    eval_file: str = Field(min_length=1, max_length=1000)
    output_dir: str = Field(min_length=1, max_length=1000)
    device: Literal["auto", "cuda", "mps", "cpu"]
    precision: Literal["bf16", "fp16", "fp32"]
    batch_size: int = Field(ge=1, le=256)
    grad_acc: int = Field(ge=1, le=1024)
    learning_rate: float = Field(gt=0, le=1)
    epochs: int = Field(ge=1, le=100)
    save_steps: int = Field(ge=1, le=1_000_000)
    save_total_limit: int = Field(ge=1, le=100)
    num_workers: int = Field(ge=0, le=64)
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int = Field(ge=1, le=32)
    resume_from: str = Field(default="", max_length=1000)
    resume_latest: bool = False

    def to_config(self) -> Qwen3TrainingConfig:
        return Qwen3TrainingConfig.from_mapping(self.model_dump())
