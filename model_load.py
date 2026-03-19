import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    AutoModel,
    SeamlessM4Tv2ForSpeechToText,
    VibeVoiceAsrForConditionalGeneration,
)
import evaluate

# =============================================================================
# Models compatible with finetune_whisper.py (standard Whisper models)
# =============================================================================

# Uncomment any of these to fine-tune with finetune_whisper.py:
# model_name = "openai/whisper-base"
# model_name = "openai/whisper-medium"
# model_name = "openai/whisper-large-v3"
# model_name = "openai/whisper-large-v3-turbo"

# =============================================================================
# Models that are NOT compatible with finetune_whisper.py
# They require different loading and fine-tuning approaches.
# =============================================================================

# --- microsoft/VibeVoice-ASR (9B params) ---
# HF-native: uses VibeVoiceAsrForConditionalGeneration + AutoProcessor (transformers >= 5.3.0)
print("Loading microsoft/VibeVoice-ASR")
processor6 = AutoProcessor.from_pretrained("microsoft/VibeVoice-ASR")
model6 = VibeVoiceAsrForConditionalGeneration.from_pretrained(
    "microsoft/VibeVoice-ASR",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print("VibeVoice-ASR loaded successfully")

# --- Systran/faster-whisper-large-v3 ---
# CTranslate2 format — cannot be loaded with HuggingFace Transformers.
# Use: pip install faster-whisper
# from faster_whisper import WhisperModel
# model7 = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("Skipping Systran/faster-whisper-large-v3 (CTranslate2 format, not trainable)")

# --- facebook/seamless-m4t-v2-large (2.3B params) ---
print("Loading facebook/seamless-m4t-v2-large")
tokenizer8 = AutoTokenizer.from_pretrained("facebook/seamless-m4t-v2-large")
model8 = AutoModel.from_pretrained("facebook/seamless-m4t-v2-large")
print("SeamlessM4T v2 loaded successfully")

wer = evaluate.load("wer")
