"""
Inference script using a Whisper model from HuggingFace Hub.
Transcribes all audio files in a directory and outputs a CSV (path,sentence).

Usage:
    # Use a public model from HuggingFace
    python inference_whisper_hf.py --model openai/whisper-large-v3 --test_dir test --output results.csv

    # Use a private/gated model with HuggingFace token
    python inference_whisper_hf.py --model your-username/whisper-thai-finetuned --hf_token hf_xxxxx --test_dir test

    # Use a specific model from HuggingFace community
    python inference_whisper_hf.py --model biodatlab/whisper-th-medium-combined --test_dir test
"""

import argparse
import csv
from pathlib import Path

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using Whisper model from HuggingFace Hub")
    parser.add_argument("--model", default="openai/whisper-large-v3", help="HuggingFace model ID (e.g. openai/whisper-large-v3)")
    parser.add_argument("--hf_token", default=None, help="HuggingFace token for private/gated models")
    parser.add_argument("--test_dir", default="test", help="Directory containing audio files")
    parser.add_argument("--output", default="results_hg.csv", help="Output CSV file path")
    parser.add_argument("--language", default="th", help="Language code (default: th)")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    audio_extensions = (".mp3", ".wav", ".flac", ".ogg", ".m4a")

    print(f"Loading model from HuggingFace: {args.model}...")
    processor = WhisperProcessor.from_pretrained(args.model, token=args.hf_token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(args.model, token=args.hf_token, torch_dtype=dtype)
    model.to(device)
    model.eval()
    print(f"Using device: {device}")

    audio_files = sorted(f for f in test_dir.iterdir() if f.suffix.lower() in audio_extensions)
    print(f"Found {len(audio_files)} audio files in {test_dir}")

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "sentence"])

        for i, audio_file in enumerate(audio_files, 1):
            try:
                audio, _ = librosa.load(str(audio_file), sr=16000)
                inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype=dtype)

                with torch.no_grad():
                    predicted_ids = model.generate(inputs, language=args.language, task="transcribe")

                text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
                writer.writerow([audio_file.name, text])
                print(f"[{i}/{len(audio_files)}] Transcribing: {audio_file.name}")
                print(f"  -> {text}")
            except Exception as e:
                print(f"[{i}/{len(audio_files)}] Transcribing: {audio_file.name}")
                print(f"  -> ERROR: {e}")
                writer.writerow([audio_file.name, ""])

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
