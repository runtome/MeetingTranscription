"""
Fine-tune Facebook SeamlessM4T v2 Large for Thai Speech Recognition.

Uses HuggingFace Transformers with SeamlessM4Tv2ForSpeechToText.
Expects training data in CSV format (path,sentence) with corresponding audio files.

Usage:
    python finetune_seamless.py
    python finetune_seamless.py --model_name facebook/seamless-m4t-v2-large --batch_size 2 --epochs 3
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune SeamlessM4T v2 for Thai speech recognition"
    )
    parser.add_argument(
        "--model_name", default="facebook/seamless-m4t-v2-large",
        help="HuggingFace model name (default: facebook/seamless-m4t-v2-large)"
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
        "--output_dir", default="./seamless-thai-finetuned",
        help="Output directory for fine-tuned model (default: ./seamless-thai-finetuned)"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size (default: 2)")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps (default: 500)")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps (default: 500)")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps (default: 500)")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=2,
        help="Gradient accumulation steps (default: 2)"
    )
    parser.add_argument("--tgt_lang", default="tha", help="Target language code (default: tha)")
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
        AutoProcessor,
        SeamlessM4Tv2ForSpeechToText,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    class SeamlessSpeechDataset(torch.utils.data.Dataset):
        """Dataset that reads CSV (path,sentence) and loads audio for SeamlessM4T v2."""

        def __init__(self, csv_path, audio_dir, processor, tgt_lang="tha"):
            self.audio_dir = Path(audio_dir)
            self.processor = processor
            self.tgt_lang = tgt_lang
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
                audio = np.zeros(16000, dtype=np.float32)
                sentence = ""

            # Process audio inputs — SeamlessM4T expects 16kHz mono audio
            audio_inputs = self.processor(
                audios=audio, sampling_rate=16000, return_tensors="np"
            )

            # Tokenize target text using the text tokenizer with target language
            text_inputs = self.processor.tokenizer(
                text=sentence,
                src_lang=self.tgt_lang,
                return_tensors="np",
            )
            labels = text_inputs["input_ids"][0]

            return {
                "input_features": audio_inputs["input_features"][0],
                "attention_mask": audio_inputs.get("attention_mask", np.ones(audio_inputs["input_features"].shape[1:2]))[0] if "attention_mask" in audio_inputs else None,
                "labels": labels,
            }

    @dataclass
    class DataCollatorSeamlessSeq2Seq:
        """Pads input features and labels for SeamlessM4T v2."""

        processor: object

        def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, torch.Tensor]:
            # Pad audio input features
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            # Pad labels
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            batch["labels"] = labels
            return batch

    def compute_metrics(pred):
        """Decode predictions/labels and compute WER."""
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = [
            processor.decode(ids.tolist(), skip_special_tokens=True)
            for ids in pred_ids
        ]
        label_str = [
            processor.decode(ids.tolist(), skip_special_tokens=True)
            for ids in label_ids
        ]

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": 100 * wer}

    print(f"Loading processor and model: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        args.model_name, torch_dtype=torch.float32
    )

    print("Loading datasets...")
    train_dataset = SeamlessSpeechDataset(
        args.train_csv, args.train_audio_dir, processor, tgt_lang=args.tgt_lang
    )
    val_dataset = SeamlessSpeechDataset(
        args.val_csv, args.val_audio_dir, processor, tgt_lang=args.tgt_lang
    )

    data_collator = DataCollatorSeamlessSeq2Seq(processor=processor)
    wer_metric = evaluate.load("wer")

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

    print("Done! Fine-tuned SeamlessM4T v2 model saved.")


if __name__ == "__main__":
    main()
