"""
Batch Transcription to CSV
Transcribe all .mp3 files in the test folder and output results to submission.csv

Supports both openai-whisper and fine-tuned HuggingFace models.
"""

import argparse
import csv
from pathlib import Path


def batch_transcribe_to_csv(input_dir="test", output_file="submission.csv", model_size="medium"):
    import whisper

    input_path = Path(input_dir)
    audio_files = sorted(input_path.glob("*.mp3"))

    if not audio_files:
        print(f"No .mp3 files found in {input_dir}")
        return

    print(f"Found {len(audio_files)} .mp3 files")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available, running on CPU (this will be very slow!)")

    print(f"Loading Whisper {model_size} model...")
    model = whisper.load_model(model_size, device=device)

    results = []
    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] Transcribing: {audio_file.name}")
        result = model.transcribe(str(audio_file), language="th", fp16=(device == "cuda"))
        sentence = result["text"].strip()
        results.append((audio_file.name, sentence))
        print(f"  -> {sentence[:80]}{'...' if len(sentence) > 80 else ''}")

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "sentence"])
        writer.writerows(results)

    print(f"\nDone! Results saved to {output_file} ({len(results)} files)")


def batch_transcribe_with_finetuned(model_path, input_dir="test", output_file="submission.csv"):
    """Transcribe using a fine-tuned HuggingFace Whisper model."""
    from transformers import pipeline

    input_path = Path(input_dir)
    audio_files = sorted(input_path.glob("*.mp3"))

    if not audio_files:
        print(f"No .mp3 files found in {input_dir}")
        return

    print(f"Found {len(audio_files)} .mp3 files")
    print(f"Loading fine-tuned model from {model_path}...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device="cuda:0" if __import__("torch").cuda.is_available() else "cpu",
    )

    results = []
    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] Transcribing: {audio_file.name}")
        result = pipe(str(audio_file), generate_kwargs={"language": "th", "task": "transcribe"})
        sentence = result["text"].strip()
        results.append((audio_file.name, sentence))
        print(f"  -> {sentence[:80]}{'...' if len(sentence) > 80 else ''}")

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "sentence"])
        writer.writerows(results)

    print(f"\nDone! Results saved to {output_file} ({len(results)} files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch transcribe MP3 files to CSV")
    parser.add_argument("--input_dir", default="test", help="Directory with .mp3 files (default: test)")
    parser.add_argument("--output_file", default="submission.csv", help="Output CSV file (default: submission.csv)")
    parser.add_argument("--model_size", default="medium", help="Whisper model size (default: medium)")
    parser.add_argument(
        "--finetuned_model", default=None,
        help="Path to fine-tuned HuggingFace model (uses transformers pipeline instead of openai-whisper)"
    )
    args = parser.parse_args()

    if args.finetuned_model:
        batch_transcribe_with_finetuned(args.finetuned_model, args.input_dir, args.output_file)
    else:
        batch_transcribe_to_csv(args.input_dir, args.output_file, args.model_size)
