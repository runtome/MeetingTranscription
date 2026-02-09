# Quick Setup Guide - Thai Meeting Transcription

## 1. Prerequisites

Install Python 3.8+ and FFmpeg:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv ffmpeg

# macOS (requires Homebrew)
brew install python ffmpeg
```

## 2. Setup Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
```

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will download and install:
- OpenAI Whisper (~1.5GB for medium model on first run)
- PyTorch with CUDA support (if GPU available)
- Pyannote Audio for speaker diarization
- Audio processing libraries

## 4. Test Installation

```bash
python test_installation.py
```

This will verify all components are working correctly.

## 5. Quick Start Examples

### Example 1: Simple transcription (no speakers)

```bash
# Edit simple_transcribe.py and set your audio file path
python simple_transcribe.py
```

### Example 2: Command line with speaker diarization

```bash
# First, get HuggingFace token:
# 1. Sign up at https://huggingface.co
# 2. Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1
# 3. Get token: https://huggingface.co/settings/tokens

python transcribe_meeting.py meeting.mp3 --hf-token YOUR_TOKEN
```

### Example 3: Without speaker diarization (faster)

```bash
python transcribe_meeting.py meeting.mp3 --no-speakers
```

## 6. Useful Commands

### Get audio file info
```bash
python audio_utils.py info meeting.mp3
```

### Convert to WAV
```bash
python audio_utils.py convert meeting.mp3
```

### Split long audio
```bash
python audio_utils.py split long_meeting.mp3 -l 15  # 15-minute segments
```

### Batch process multiple files
```bash
python batch_transcribe.py ./audio_folder -o ./output --hf-token YOUR_TOKEN
```

## 7. Output Files

Each transcription creates 3 files:
- `filename_transcript.txt` - Human-readable with timestamps
- `filename_transcript.json` - Machine-readable JSON format
- `filename_transcript.srt` - Subtitle format for video

## 8. Tips

- **Start with small model** for testing: `-m small`
- **Use GPU** for faster processing (10-20x speedup)
- **For production**: Use `-m medium` or `-m large`
- **Long meetings**: Split audio into 10-15 minute chunks first

## Troubleshooting

### "FFmpeg not found"
Install FFmpeg (see step 1)

### "CUDA out of memory"
Use smaller model: `-m small` or `-m base`

### "Module not found"
Make sure virtual environment is activated and dependencies installed

### Speaker diarization errors
- Ensure you have HuggingFace account
- Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
- Use valid access token

## Next Steps

Read `README.md` for complete documentation including:
- All command line options
- Python API usage
- Advanced configuration
- Performance benchmarks
- Best practices for audio quality
