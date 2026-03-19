"""
Fine-tune Microsoft VibeVoice-ASR for Thai Speech Recognition using LoRA.

Requires: transformers >= 5.3.0, peft
    pip install --upgrade transformers>=5.3.0
    pip install peft

Uses training data in CSV format (path,sentence) with corresponding audio files.

Usage:
    python finetune_vibe_voice.py
    python finetune_vibe_voice.py --model_name microsoft/VibeVoice-ASR --batch_size 1 --epochs 3
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune VibeVoice-ASR for Thai speech recognition (LoRA)"
    )
    parser.add_argument(
        "--model_name", default="microsoft/VibeVoice-ASR",
        help="HuggingFace model name (default: microsoft/VibeVoice-ASR)"
    )
    parser.add_argument(
        "--train_csv", default="train/annotation/train.csv",
        help="Path to training CSV (default: train/annotation/train.csv)"
    )
    parser.add_argument(
        "--train_audio_dir", default="train/audio",
        help="Directory with training audio files (default: train/audio)"
    )
    parser.add_argument(
        "--val_csv", default="val/annotation/dev.csv",
        help="Path to validation CSV (default: val/annotation/dev.csv)"
    )
    parser.add_argument(
        "--val_audio_dir", default="val/audio",
        help="Directory with validation audio files (default: val/audio)"
    )
    parser.add_argument(
        "--output_dir", default="./vibevoice-thai-finetuned",
        help="Output directory for fine-tuned model (default: ./vibevoice-thai-finetuned)"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size (default: 1)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio (default: 0.1)")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps (default: 100)")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps (default: 10)")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4,
        help="Gradient accumulation steps (default: 4)"
    )
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)")
    args = parser.parse_args()

    import csv
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Any, Dict, List

    import librosa
    import numpy as np
    import torch
    from transformers import (
        AutoProcessor,
        VibeVoiceAsrForConditionalGeneration,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    class VibeVoiceDataset(torch.utils.data.Dataset):
        """Dataset that reads CSV (path,sentence) and prepares inputs for VibeVoice-ASR.

        Uses the HF-native chat template format for training.
        """

        def __init__(self, csv_path, audio_dir, processor):
            self.audio_dir = Path(audio_dir)
            self.processor = processor
            self.samples = []

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.samples.append((row["path"], row["sentence"]))

            print(f"Loaded {len(self.samples)} samples from {csv_path}")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            filename, sentence = self.samples[idx]
            audio_path = str(self.audio_dir / filename)

            try:
                audio, _ = librosa.load(audio_path, sr=16000)
            except Exception as e:
                print(f"WARNING: Skipping corrupted file {filename}: {e}")
                audio = np.zeros(16000, dtype=np.float32)
                sentence = ""

            # Use apply_chat_template with audio + target text for training
            chat_template = [[{
                "role": "user",
                "content": [
                    {"type": "text", "text": sentence},
                    {"type": "audio", "audio": audio},
                ],
            }]]
            inputs = self.processor.apply_chat_template(
                chat_template,
                tokenize=True,
                return_dict=True,
                output_labels=True,
            )
            return {k: v.squeeze(0) if hasattr(v, "squeeze") else v for k, v in inputs.items()}

    @dataclass
    class VibeVoiceDataCollator:
        """Pads input features and labels for VibeVoice-ASR."""

        processor: object
        label_pad_token_id: int = -100

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            batch = {}
            keys = features[0].keys()

            for key in keys:
                values = [f[key] for f in features]
                if isinstance(values[0], torch.Tensor):
                    max_len = max(v.shape[-1] for v in values)
                    pad_value = self.label_pad_token_id if key == "labels" else 0
                    padded = []
                    for v in values:
                        pad_size = max_len - v.shape[-1]
                        if pad_size > 0:
                            v = torch.nn.functional.pad(v, (0, pad_size), value=pad_value)
                        padded.append(v)
                    batch[key] = torch.stack(padded)
                else:
                    batch[key] = values

            return batch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading processor and model: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto",
    )

    # Apply LoRA (9B model — full fine-tuning is impractical)
    print("Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading datasets...")
    train_dataset = VibeVoiceDataset(args.train_csv, args.train_audio_dir, processor)
    val_dataset = VibeVoiceDataset(args.val_csv, args.val_audio_dir, processor)

    data_collator = VibeVoiceDataCollator(processor=processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("Starting LoRA fine-tuning...")
    trainer.train()

    print(f"Saving LoRA adapters to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print("Done! Fine-tuned VibeVoice-ASR LoRA adapters saved.")


if __name__ == "__main__":
    main()
