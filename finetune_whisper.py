"""
Fine-tune OpenAI Whisper for Thai Speech Recognition

Uses HuggingFace Transformers with WhisperForConditionalGeneration.
Expects training data in CSV format (path,sentence) with corresponding audio files.

Usage:
    python finetune_whisper.py
    python finetune_whisper.py --model_name openai/whisper-medium --batch_size 4 --epochs 5
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
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps (default: 500)")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps (default: 500)")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Gradient accumulation steps (default: 1)"
    )
    args = parser.parse_args()

    import csv
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Dict, List, Union

    import evaluate
    import librosa
    import numpy as np
    import torch
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    class ThaiSpeechDataset(torch.utils.data.Dataset):
        """Dataset that reads CSV (path,sentence) and loads audio with librosa."""

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
            audio_path = self.audio_dir / filename

            try:
                audio, _ = librosa.load(str(audio_path), sr=16000)
            except Exception as e:
                print(f"WARNING: Skipping corrupted file {filename}: {e}")
                # Return a short silence with empty transcription
                audio = np.zeros(16000, dtype=np.float32)
                sentence = ""

            input_features = self.processor(
                audio, sampling_rate=16000, return_tensors="np"
            ).input_features[0]

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

    print("Loading datasets...")
    train_dataset = ThaiSpeechDataset(args.train_csv, args.train_audio_dir, processor)
    val_dataset = ThaiSpeechDataset(args.val_csv, args.val_audio_dir, processor)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    wer_metric = evaluate.load("wer")
    # wer_metric = evaluate.load("wer", cache_dir="/project/zz991000-zdeva/zz991010/hf/evaluate")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
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

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print("Done! Fine-tuned model saved.")


if __name__ == "__main__":
    main()
