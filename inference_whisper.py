"""
Inference script for fine-tuned Whisper model.
Transcribes all audio files in a directory and outputs a CSV (path,sentence).

Usage:
    python inference_whisper.py --model ./whisper-thai-finetuned --test_dir test --output results.csv
"""

import argparse
import csv
import re
from pathlib import Path

import librosa
import torch
from pythainlp.tokenize import word_tokenize
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def clean_thai_asr_keep_space(text):
    if not text:
        return ""

    text = str(text)

    # 1. remove long character repeat
    text = re.sub(r'(.)\1{4,}', r'\1', text)

    # 2. remove repeated phrases
    text = re.sub(r'(.{2,20})\1{3,}', r'\1', text)

    # 3. remove repeated words but keep spacing
    tokens = word_tokenize(text, engine="newmm")

    result = []
    prev = None
    for t in tokens:
        if t != prev:
            result.append(t)
        prev = t

    return "".join(result)


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using fine-tuned Whisper")
    parser.add_argument("--model", default="./whisper-thai-finetuned", help="Path to fine-tuned model")
    parser.add_argument("--test_dir", default="test", help="Directory containing audio files")
    parser.add_argument("--output", default="results.csv", help="Output CSV file path")
    parser.add_argument("--language", default="th", help="Language code (default: th)")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    audio_extensions = (".mp3", ".wav", ".flac", ".ogg", ".m4a")

    print(f"Loading model from {args.model}...")
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
                inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)

                with torch.no_grad():
                    predicted_ids = model.generate(inputs, language=args.language, task="transcribe")

                text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
                cleaned_text = clean_thai_asr_keep_space(text)
                # Fall back to raw text if cleaning produced empty string
                final_text = cleaned_text if cleaned_text.strip() else text
                writer.writerow([audio_file.name, final_text])
                print(f"[{i}/{len(audio_files)}] Transcribing: {audio_file.name}")
                print(f"  -> raw:     {text}")
                print(f"  -> cleaned: {cleaned_text}")
            except Exception as e:
                print(f"[{i}/{len(audio_files)}] Transcribing: {audio_file.name}")
                print(f"  -> ERROR: {e}")
                writer.writerow([audio_file.name, ""])

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
