from transformers import WhisperProcessor, WhisperForConditionalGeneration

model_name = "openai/whisper-base"

processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
