"""使用 mlx-tune 在独立进程中训练 Qwen3-ASR LoRA 适配器。"""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_tune import FastSTTModel, STTDataCollator, STTSFTConfig, STTSFTTrainer

from app.config import Qwen3TrainingConfig


def load_dataset(path: str) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for line_number, line in enumerate(Path(path).expanduser().read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        audio_path = Path(str(item["audio"])).expanduser()
        text = str(item["text"]).strip()
        if not audio_path.is_file() or not text:
            raise ValueError(f"invalid training sample at {path}:{line_number}")
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        samples.append({"audio": {"array": mx.array(audio), "sampling_rate": sample_rate}, "text": text})
    if not samples:
        raise ValueError(f"training dataset is empty: {path}")
    return samples


def run_training(config: Qwen3TrainingConfig) -> None:
    train_dataset = load_dataset(config.train_file)
    eval_dataset = load_dataset(config.eval_file) if Path(config.eval_file).expanduser().is_file() else None
    model, processor = FastSTTModel.from_pretrained(config.model_path, max_seq_length=448)
    model = FastSTTModel.get_peft_model(
        model, r=16, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        finetune_encoder=True, finetune_decoder=True,
    )
    collator = STTDataCollator(model=model, processor=processor, language="Chinese", task="transcribe")
    requested_args = {
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.grad_acc,
        "learning_rate": config.learning_rate,
        "max_steps": max(1, math.ceil(len(train_dataset) * config.epochs / config.batch_size / config.grad_acc)),
        "logging_steps": 1,
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "output_dir": str(Path(config.output_dir).expanduser()),
        "remove_unused_columns": False,
        "dataloader_drop_last": False,
    }
    supported = set(inspect.signature(STTSFTConfig).parameters)
    trainer_args = STTSFTConfig(**{key: value for key, value in requested_args.items() if key in supported})
    trainer = STTSFTTrainer(
        model=model, processor=processor, data_collator=collator,
        train_dataset=train_dataset, eval_dataset=eval_dataset, args=trainer_args,
    )
    train_signature = inspect.signature(trainer.train)
    if config.resume_from and "resume_from_checkpoint" in train_signature.parameters:
        trainer.train(resume_from_checkpoint=config.resume_from)
    else:
        trainer.train()
    model.save_pretrained(str(Path(config.output_dir).expanduser() / "adapters"))
