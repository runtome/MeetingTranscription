# Configuration file for Thai Meeting Transcription

# Whisper Settings
WHISPER_MODEL = "medium"  # Options: tiny, base, small, medium, large
LANGUAGE = "th"  # Thai language code

# Speaker Diarization Settings
ENABLE_SPEAKER_DIARIZATION = True
MIN_SPEAKERS = 2  # Minimum number of speakers expected
MAX_SPEAKERS = 10  # Maximum number of speakers expected

# HuggingFace Token (required for speaker diarization)
# Get your token from: https://huggingface.co/settings/tokens
# Accept model terms: https://huggingface.co/pyannote/speaker-diarization-3.1
HF_TOKEN = "your_huggingface_token_here"

# Output Settings
OUTPUT_DIRECTORY = "./transcriptions"
OUTPUT_FORMATS = ["txt", "json", "srt"]  # Available formats

# Audio Processing
# Supported formats: mp3, wav, m4a, flac, ogg, etc.
SUPPORTED_FORMATS = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma"]

# Performance Settings
USE_GPU = True  # Set to False to use CPU only
BATCH_SIZE = 16  # Batch size for processing (adjust based on GPU memory)

# Advanced Whisper Settings
WHISPER_TEMPERATURE = 0  # Sampling temperature (0 = deterministic)
WHISPER_BEAM_SIZE = 5  # Beam search size
WHISPER_BEST_OF = 5  # Number of candidates to consider

# Logging
VERBOSE = True  # Print detailed progress
LOG_FILE = "transcription.log"
