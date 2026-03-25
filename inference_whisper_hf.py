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
    parser = argparse.ArgumentParser(description="Transcribe audio files using Whisper model from HuggingFace Hub")
    parser.add_argument("--model", default="openai/whisper-large-v3", help="HuggingFace model ID (e.g. openai/whisper-large-v3)")
    parser.add_argument("--hf_token", default=None, help="HuggingFace token for private/gated models")
    parser.add_argument("--test_dir", default="test", help="Directory containing audio files")
    parser.add_argument("--output", default="results_hg.csv", help="Output CSV file path")
    parser.add_argument("--language", default="th", help="Language code (default: th)")
    parser.add_argument("--num_beams", type=int, default=1, help="Beam search width (default: 1=greedy, try 5 for beam search)")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty (default: 1.0=off, try 1.1-1.2)")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0, help="Block repeating n-grams of this size (default: 0=off)")
    parser.add_argument("--clean", action="store_true", help="Enable ASR text cleaning (default: off)")
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
                    gen_kwargs = dict(
                        language=args.language,
                        task="transcribe",
                        num_beams=args.num_beams,
                    )
                    if args.repetition_penalty != 1.0:
                        gen_kwargs["repetition_penalty"] = args.repetition_penalty
                    if args.no_repeat_ngram_size > 0:
                        gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
                    predicted_ids = model.generate(inputs, **gen_kwargs)

                text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
                output_text = clean_thai_asr_keep_space(text) if args.clean else text
                writer.writerow([audio_file.name, output_text])
                print(f"[{i}/{len(audio_files)}] Transcribing: {audio_file.name}")
                print(f"  -> raw:     {text}")
                if args.clean:
                    print(f"  -> cleaned: {output_text}")
            except Exception as e:
                print(f"[{i}/{len(audio_files)}] Transcribing: {audio_file.name}")
                print(f"  -> ERROR: {e}")
                writer.writerow([audio_file.name, ""])

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
