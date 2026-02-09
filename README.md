# Thai Meeting Transcription Tool

Automatic transcription tool for Thai language meetings using OpenAI Whisper with optional speaker diarization (speaker separation).

## Features

- ✅ **Accurate Thai transcription** using OpenAI Whisper
- ✅ **Speaker diarization** - Automatically separate different speakers
- ✅ **Multiple output formats** - TXT, JSON, SRT (subtitle format)
- ✅ **Timestamps** - Each segment includes timing information
- ✅ **GPU acceleration** - Faster processing with CUDA support
- ✅ **Multiple audio formats** - Supports MP3, WAV, M4A, FLAC, etc.

## Requirements

- Python 3.8 or higher
- FFmpeg (for audio processing)
- CUDA-capable GPU (optional, but recommended for faster processing)

## Installation

### 1. Install FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate on Linux/Mac:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Note:** The first time you run the tool, it will download the Whisper model (~1.5GB for medium model).

### 4. Setup for Speaker Diarization (Optional)

Speaker diarization requires a HuggingFace account:

1. Create account at https://huggingface.co/join
2. Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Get your access token from https://huggingface.co/settings/tokens
4. Save token in `config.py` or pass via command line

## Quick Start

### Simple Transcription (No Speaker Labels)

```bash
python simple_transcribe.py
```

Edit `simple_transcribe.py` and change the `audio_file` variable to your file path.

### Command Line Usage

**Basic transcription without speakers:**
```bash
python transcribe_meeting.py meeting.mp3 --no-speakers
```

**With speaker diarization:**
```bash
python transcribe_meeting.py meeting.mp3 --hf-token YOUR_TOKEN
```

**Specify output directory:**
```bash
python transcribe_meeting.py meeting.mp3 -o ./output --hf-token YOUR_TOKEN
```

**Choose different model size:**
```bash
python transcribe_meeting.py meeting.mp3 -m large --hf-token YOUR_TOKEN
```

## Command Line Options

```
usage: transcribe_meeting.py [-h] [-o OUTPUT] [-m MODEL] [-l LANGUAGE] 
                             [--no-speakers] [--hf-token HF_TOKEN] 
                             audio_file

Arguments:
  audio_file            Path to audio file (mp3, wav, m4a, etc.)

Options:
  -h, --help            Show help message
  -o, --output OUTPUT   Output directory (default: ./transcriptions)
  -m, --model MODEL     Whisper model size: tiny, base, small, medium, large
                        (default: medium)
  -l, --language LANGUAGE
                        Language code (default: th for Thai)
  --no-speakers         Disable speaker diarization (faster)
  --hf-token HF_TOKEN   HuggingFace access token (for speaker diarization)
```

## Model Sizes

| Model  | Parameters | Speed | Accuracy | VRAM Required |
|--------|-----------|-------|----------|---------------|
| tiny   | 39M       | ~32x  | Low      | ~1 GB         |
| base   | 74M       | ~16x  | Fair     | ~1 GB         |
| small  | 244M      | ~6x   | Good     | ~2 GB         |
| medium | 769M      | ~2x   | Better   | ~5 GB         |
| large  | 1550M     | 1x    | Best     | ~10 GB        |

**Recommendation:** 
- For quick testing: `base` or `small`
- For production: `medium` or `large`
- Thai language works well with `medium` model

## Output Formats

The tool generates three output files:

### 1. TXT Format (Human-readable)
```
[SPEAKER_00]
[00:00:05] สวัสดีครับ วันนี้เราจะมาประชุมเรื่องโปรเจคใหม่

[SPEAKER_01]
[00:00:12] ขอบคุณครับ ผมมีข้อเสนอเกี่ยวกับแผนการตลาด
```

### 2. JSON Format (For processing)
```json
[
  {
    "start": 5.0,
    "end": 12.0,
    "speaker": "SPEAKER_00",
    "text": "สวัสดีครับ วันนี้เราจะมาประชุมเรื่องโปรเจคใหม่"
  },
  {
    "start": 12.0,
    "end": 18.5,
    "speaker": "SPEAKER_01",
    "text": "ขอบคุณครับ ผมมีข้อเสนอเกี่ยวกับแผนการตลาด"
  }
]
```

### 3. SRT Format (Subtitle format)
```
1
00:00:05,000 --> 00:00:12,000
[SPEAKER_00] สวัสดีครับ วันนี้เราจะมาประชุมเรื่องโปรเจคใหม่

2
00:00:12,000 --> 00:00:18,500
[SPEAKER_01] ขอบคุณครับ ผมมีข้อเสนอเกี่ยวกับแผนการตลาด
```

## Python API Usage

```python
from transcribe_meeting import MeetingTranscriber

# Initialize
transcriber = MeetingTranscriber(
    whisper_model="medium",
    language="th"
)

# Transcribe with speakers
transcriber.process_meeting(
    audio_path="meeting.mp3",
    output_dir="./output",
    with_speakers=True,
    hf_token="your_hf_token"
)

# Or just transcribe without speakers (faster)
transcriber.process_meeting(
    audio_path="meeting.mp3",
    output_dir="./output",
    with_speakers=False
)
```

## Tips for Best Results

### Audio Quality
- Use clear audio with minimal background noise
- Recommended format: WAV or FLAC (lossless)
- Sample rate: 16kHz or higher
- Mono or stereo both work fine

### Speaker Diarization
- Works best with 2-6 speakers
- Each speaker should speak for at least 2-3 seconds per turn
- Minimize speaker overlap (people talking simultaneously)
- Clear separation between speakers improves accuracy

### Performance Optimization
- Use GPU if available (10-20x faster than CPU)
- For long meetings (>1 hour), consider splitting into smaller chunks
- Lower model size for faster processing at cost of accuracy

## Troubleshooting

### Common Issues

**1. FFmpeg not found**
```
Error: ffmpeg not found
```
Solution: Install FFmpeg (see Installation section)

**2. CUDA out of memory**
```
RuntimeError: CUDA out of memory
```
Solution: Use smaller model (`-m small` or `-m base`) or process on CPU

**3. HuggingFace token error**
```
Error: pyannote model requires authentication
```
Solution: 
- Get token from https://huggingface.co/settings/tokens
- Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1

**4. Module not found**
```
ModuleNotFoundError: No module named 'whisper'
```
Solution: Make sure you activated virtual environment and ran `pip install -r requirements.txt`

## Project Structure

```
thai-transcription/
├── transcribe_meeting.py    # Main script with all features
├── simple_transcribe.py     # Simple example for quick start
├── config.py               # Configuration file
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── transcriptions/        # Output directory (created automatically)
    ├── meeting_transcript.txt
    ├── meeting_transcript.json
    └── meeting_transcript.srt
```

## Examples

### Example 1: Quick transcription without speakers
```bash
python transcribe_meeting.py interview.mp3 --no-speakers -o ./transcripts
```

### Example 2: Full meeting with speaker labels
```bash
python transcribe_meeting.py meeting.m4a \
    --hf-token hf_xxxxxxxxxxxx \
    -o ./output \
    -m large
```

### Example 3: Using the simple script
```python
# Edit simple_transcribe.py
audio_file = "path/to/your/meeting.mp3"

# Run
python simple_transcribe.py
```

## Advanced Configuration

Edit `config.py` to customize:
- Default model size
- Number of expected speakers
- Output formats
- GPU/CPU usage
- Whisper advanced parameters

## Performance Benchmarks

Processing time for a 30-minute meeting:

| Model  | GPU (RTX 3090) | CPU (i7-12700K) |
|--------|---------------|-----------------|
| tiny   | ~2 min        | ~15 min         |
| base   | ~3 min        | ~25 min         |
| small  | ~5 min        | ~45 min         |
| medium | ~8 min        | ~90 min         |
| large  | ~15 min       | ~180 min        |

*Note: Times include both transcription and speaker diarization*

## Supported Languages

While optimized for Thai (`th`), this tool supports 99+ languages:
- Thai (th)
- English (en)
- Chinese (zh)
- Japanese (ja)
- Korean (ko)
- And many more...

Change language with `-l` flag:
```bash
python transcribe_meeting.py meeting.mp3 -l en  # English
```

## Credits

- **OpenAI Whisper**: https://github.com/openai/whisper
- **Pyannote Audio**: https://github.com/pyannote/pyannote-audio

## License

This project uses:
- OpenAI Whisper (MIT License)
- Pyannote Audio (MIT License)

## Contributing

Feel free to submit issues, feature requests, or pull requests!

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review OpenAI Whisper documentation: https://github.com/openai/whisper
3. Review Pyannote documentation: https://github.com/pyannote/pyannote-audio
