from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate

model_name = "openai/whisper-base"

processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

wer = evaluate.load("wer")
