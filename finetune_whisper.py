"""
Fine-tune OpenAI Whisper for Thai Speech Recognition

Uses HuggingFace Transformers with WhisperForConditionalGeneration.
Expects training data in CSV format (path,sentence) with corresponding audio files.

Usage:
    python finetune_whisper.py
    python finetune_whisper.py --model_name openai/whisper-medium --batch_size 4 --epochs 5

    # T13 recommended: T6 recipe + augmentation (1e-5 LR proven best for typhoon)
    python finetune_whisper.py --model_name typhoon-ai/typhoon-whisper-large-v3 --epochs 10 --learning_rate 1e-5 --augment --encoder_lr_factor 1.0

    # Optional: freeze encoder (only decoder trains)
    python finetune_whisper.py --model_name typhoon-ai/typhoon-whisper-large-v3 --epochs 10 --learning_rate 1e-5 --augment --freeze_encoder
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper for Thai speech recognition"
    )
    parser.add_argument(
        "--model_name", default="openai/whisper-base",
        help="HuggingFace model name (default: openai/whisper-base)"
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
        "--output_dir", default="./whisper-thai-finetuned",
        help="Output directory for fine-tuned model (default: ./whisper-thai-finetuned)"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size (default: 8)")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps (default: 500)")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every N steps (default: 200)")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluate every N steps (default: 200)")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4,
        help="Gradient accumulation steps (default: 4, effective batch = batch_size * 4)"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay (default: 0.01)")
    parser.add_argument(
        "--lr_scheduler", default="cosine",
        help="LR scheduler type: cosine, linear, constant_with_warmup (default: cosine)"
    )
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation (speed perturbation + spec augment)")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Max checkpoints to keep (default: 3)")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience in eval steps (default: 5)")
    parser.add_argument(
        "--freeze_encoder", action="store_true",
        help="Freeze encoder weights (recommended for pre-trained Thai models like typhoon-whisper)"
    )
    parser.add_argument(
        "--encoder_lr_factor", type=float, default=1.0,
        help="Multiply encoder LR by this factor vs decoder (default: 1.0=same LR, use <1.0 for differential LR)"
    )
    args = parser.parse_args()

    import csv
    import random
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Dict, List, Union

    import evaluate
    import librosa
    import numpy as np
    import torch
    from transformers import (
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    def spec_augment(mel, freq_mask_num=2, freq_mask_width=27, time_mask_num=2, time_mask_width=40):
        """Apply SpecAugment (frequency and time masking) to mel spectrogram."""
        augmented = mel.copy()
        n_freq, n_time = augmented.shape

        for _ in range(freq_mask_num):
            f = random.randint(0, min(freq_mask_width, n_freq - 1))
            f0 = random.randint(0, n_freq - f)
            augmented[f0:f0 + f, :] = 0

        for _ in range(time_mask_num):
            t = random.randint(0, min(time_mask_width, n_time - 1))
            t0 = random.randint(0, n_time - t)
            augmented[:, t0:t0 + t] = 0

        return augmented

    class ThaiSpeechDataset(torch.utils.data.Dataset):
        """Dataset that reads CSV (path,sentence) and loads audio with librosa."""

        def __init__(self, csv_path, audio_dir, processor, augment=False):
            self.audio_dir = Path(audio_dir)
            self.processor = processor
            self.augment = augment
            self.samples = []

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.samples.append((row["path"], row["sentence"]))

            print(f"Loaded {len(self.samples)} samples from {csv_path} (augment={augment})")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            filename, sentence = self.samples[idx]
            audio_path = self.audio_dir / filename

            try:
                audio, _ = librosa.load(str(audio_path), sr=16000)
            except Exception as e:
                print(f"WARNING: Skipping corrupted file {filename}: {e}")
                audio = np.zeros(16000, dtype=np.float32)
                sentence = ""

            # Speed perturbation: randomly stretch/compress audio
            if self.augment and random.random() < 0.5:
                speed = random.choice([0.9, 0.95, 1.05, 1.1])
                audio = librosa.effects.time_stretch(audio, rate=speed)

            input_features = self.processor(
                audio, sampling_rate=16000, return_tensors="np"
            ).input_features[0]

            # SpecAugment: mask frequency and time bands
            if self.augment and random.random() < 0.5:
                input_features = spec_augment(input_features)

            labels = self.processor.tokenizer(sentence).input_ids

            # Whisper max label length is 448 tokens; truncate to avoid ValueError
            if len(labels) > 448:
                labels = labels[:448]

            return {"input_features": input_features, "labels": labels}

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        """Pads input features and labels, masking label padding with -100."""

        processor: object

        def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            # Remove BOS token if the model prepends it during generation
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    def compute_metrics(pred):
        """Decode predictions/labels and compute WER."""
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": 100 * wer}

    print(f"Loading processor and model: {args.model_name}")
    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=torch.float32)

    # Force Thai language and transcribe task
    model.generation_config.language = "th"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # Freeze encoder if requested (recommended for pre-trained Thai models)
    if args.freeze_encoder:
        model.freeze_encoder()
        print("Encoder frozen — only decoder will be trained")

    print("Loading datasets...")
    train_dataset = ThaiSpeechDataset(args.train_csv, args.train_audio_dir, processor, augment=args.augment)
    val_dataset = ThaiSpeechDataset(args.val_csv, args.val_audio_dir, processor, augment=False)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    wer_metric = evaluate.load("wer")
    # wer_metric = evaluate.load("wer", cache_dir="/project/zz991000-zdeva/zz991010/hf/evaluate")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=False,
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    # Use differential LR: lower LR for encoder, full LR for decoder
    optimizers = (None, None)
    if not args.freeze_encoder and args.encoder_lr_factor < 1.0:
        from torch.optim import AdamW as TorchAdamW
        encoder_params = [p for n, p in model.named_parameters() if "encoder" in n and p.requires_grad]
        decoder_params = [p for n, p in model.named_parameters() if "encoder" not in n and p.requires_grad]
        optimizer = TorchAdamW([
            {"params": encoder_params, "lr": args.learning_rate * args.encoder_lr_factor},
            {"params": decoder_params, "lr": args.learning_rate},
        ], weight_decay=args.weight_decay)
        optimizers = (optimizer, None)  # let trainer handle scheduler
        enc_lr = args.learning_rate * args.encoder_lr_factor
        print(f"Differential LR: encoder={enc_lr:.1e}, decoder={args.learning_rate:.1e}")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        callbacks=callbacks,
        optimizers=optimizers,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print("Done! Fine-tuned model saved.")


if __name__ == "__main__":
    main()
