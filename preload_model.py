"""
Pre-download and cache models for offline use.
Run this once before submitting batch jobs.

Usage:
    # Pre-load openai-whisper model
    python preload_model.py --model medium

    # Pre-load fine-tuned HuggingFace model
    python preload_model.py --hf_model openai/whisper-medium

    # Pre-load both
    python preload_model.py --model medium --hf_model openai/whisper-medium
"""

import argparse


def preload_whisper(model_size="medium"):
    import whisper
    print(f"Downloading openai-whisper model: {model_size} ...")
    whisper.load_model(model_size)
    print(f"Done! Whisper '{model_size}' model cached.")


def preload_hf_model(model_name):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    print(f"Downloading HuggingFace model: {model_name} ...")
    WhisperProcessor.from_pretrained(model_name)
    WhisperForConditionalGeneration.from_pretrained(model_name)
    print(f"Done! HuggingFace '{model_name}' model cached.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-download and cache models")
    parser.add_argument("--model", default=None, help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--hf_model", default=None, help="HuggingFace model name (e.g. openai/whisper-medium)")
    args = parser.parse_args()

    if not args.model and not args.hf_model:
        args.model = "medium"

    if args.model:
        preload_whisper(args.model)

    if args.hf_model:
        preload_hf_model(args.hf_model)
