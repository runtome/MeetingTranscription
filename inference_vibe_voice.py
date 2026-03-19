"""
Inference script for fine-tuned VibeVoice-ASR model.
Transcribes all audio files in a directory and outputs a CSV (path,sentence).

Requires: transformers >= 5.3.0
    pip install --upgrade transformers>=5.3.0

Usage:
    python inference_vibe_voice.py --model ./vibevoice-thai-finetuned --test_dir test --output results.csv

    # With LoRA adapter on top of base model:
    python inference_vibe_voice.py --base_model microsoft/VibeVoice-ASR --lora_path ./vibevoice-thai-finetuned --test_dir test
"""

import argparse
import csv
import json
import re
from pathlib import Path

import librosa
import torch
from pythainlp.tokenize import word_tokenize


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


def extract_text_from_segments(raw_output):
    """Extract plain text from VibeVoice structured output.

    VibeVoice returns segments with speaker, text, start, end fields.
    This extracts just the text content.
    """
    try:
        segments = json.loads(raw_output)
        if isinstance(segments, list):
            texts = []
            for seg in segments:
                if isinstance(seg, dict) and seg.get("text"):
                    texts.append(seg["text"])
                elif isinstance(seg, dict) and seg.get("Content"):
                    texts.append(seg["Content"])
            if texts:
                return " ".join(texts)
    except (json.JSONDecodeError, TypeError):
        pass
    return raw_output


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using fine-tuned VibeVoice-ASR")
    parser.add_argument("--model", default=None, help="Path to merged fine-tuned model (use if LoRA was merged)")
    parser.add_argument("--base_model", default="microsoft/VibeVoice-ASR", help="Base model name")
    parser.add_argument("--lora_path", default=None, help="Path to LoRA adapter weights (if not merged)")
    parser.add_argument("--test_dir", default="test", help="Directory containing audio files")
    parser.add_argument("--output", default="results_vibevoice.csv", help="Output CSV file path")
    parser.add_argument("--max_new_tokens", type=int, default=32768, help="Max new tokens for generation")
    args = parser.parse_args()

    from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

    test_dir = Path(args.test_dir)
    audio_extensions = (".mp3", ".wav", ".flac", ".ogg", ".m4a")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model_path = args.model if args.model else args.base_model

    print(f"Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path)

    print(f"Loading model from {model_path}...")
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
    )

    # Load LoRA adapter if specified
    if args.lora_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter from {args.lora_path}...")
        model = PeftModel.from_pretrained(model, args.lora_path)

    model.eval()
    print(f"Using device: {device}")

    audio_files = sorted(f for f in test_dir.iterdir() if f.suffix.lower() in audio_extensions)
    print(f"Found {len(audio_files)} audio files in {test_dir}")

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "sentence"])

        for i, audio_file in enumerate(audio_files, 1):
            try:
                # Use apply_transcription_request for inference
                inputs = processor.apply_transcription_request(
                    audio=str(audio_file),
                ).to(model.device, model.dtype)

                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

                # Strip input tokens to get only generated tokens
                generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
                raw_text = processor.decode(generated_ids, return_format="transcription_only")[0]

                text = extract_text_from_segments(raw_text)
                cleaned_text = clean_thai_asr_keep_space(text)
                final_text = cleaned_text if cleaned_text.strip() else text

                writer.writerow([audio_file.name, final_text])
                print(f"[{i}/{len(audio_files)}] Transcribing: {audio_file.name}")
                print(f"  -> raw:     {text[:200]}")
                print(f"  -> cleaned: {cleaned_text[:200]}")
            except Exception as e:
                print(f"[{i}/{len(audio_files)}] Transcribing: {audio_file.name}")
                print(f"  -> ERROR: {e}")
                writer.writerow([audio_file.name, ""])

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
