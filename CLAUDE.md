# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Thai meeting transcription tool using OpenAI Whisper for speech-to-text with optional speaker diarization. Supports 99+ languages but defaults to Thai (`th`). Outputs TXT, JSON, and SRT formats.

## Two Versions

There are two parallel implementations with separate dependency files:

1. **Full version** (`transcribe_meeting.py` + `requirements.txt`): Uses pyannote.audio for accurate speaker diarization. Requires a HuggingFace token and accepted model terms for `pyannote/speaker-diarization-3.1`.

2. **Simple version** (`transcribe_meeting_simple.py` + `requirements_simple.txt`): No pyannote dependency. Uses pause-based or clustering-based (scikit-learn) speaker detection instead. Less accurate but easier to set up.

## Key Commands

```bash
# Install (full version)
pip install -r requirements.txt

# Install (simple version, no pyannote/HF needed)
pip install -r requirements_simple.txt

# Verify installation
python test_installation.py

# Transcribe without speakers (fastest)
python transcribe_meeting.py meeting.mp3 --no-speakers

# Transcribe with pyannote diarization
python transcribe_meeting.py meeting.mp3 --hf-token YOUR_TOKEN

# Simple version with pause-based detection
python transcribe_meeting_simple.py meeting.mp3 --method pause

# Simple version with clustering
python transcribe_meeting_simple.py meeting.mp3 --method clustering --speakers 3

# Batch processing
python batch_transcribe.py ./audio_dir/ -o ./output/

# Audio utilities (convert, normalize, split, info)
python audio_utils.py convert input.mp3
python audio_utils.py split long_meeting.mp3 -l 10
python audio_utils.py info input.mp3
```

## Architecture

- **`transcribe_meeting.py`** - `MeetingTranscriber` class: full pipeline with Whisper transcription + pyannote diarization. Merges speaker segments with transcription by calculating time overlap.
- **`transcribe_meeting_simple.py`** - `SimpleMeetingTranscriber` class: alternative pipeline using basic audio feature extraction (energy, zero-crossing rate, spectral features, pitch) + AgglomerativeClustering, or pause-based speaker change detection.
- **`batch_transcribe.py`** - Wraps `MeetingTranscriber` to process all audio files in a directory. Initializes the model once and reuses it across files.
- **`audio_utils.py`** - `AudioPreprocessor` class with CLI subcommands for convert/normalize/split/info. Uses pydub.
- **`config.py`** - Global configuration constants (model size, HF token, speaker limits, Whisper parameters). Not imported by scripts automatically; meant for user reference/customization.
- **`simple_transcribe.py`** - Minimal standalone example, no speaker labels.

## External Dependencies

- **FFmpeg** must be installed on the system (used by Whisper and pydub for audio decoding)
- **CUDA GPU** optional but recommended (auto-detected via `torch.cuda.is_available()`)
- Whisper models are downloaded on first use (~1.5GB for medium)

## Important Patterns

- All transcriber classes auto-detect CUDA/CPU device in `__init__`
- Speaker diarization merging uses max-overlap matching between Whisper segments and pyannote speaker turns
- Output files are named `{audio_stem}_transcript.{txt,json,srt}` in the specified output directory
- The `.gitignore` excludes `*.mp3` and `transcriptions/`
