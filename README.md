# Thai Meeting Transcription Tool

Automatic transcription tool for Thai language meetings using OpenAI Whisper with optional speaker diarization (speaker separation).

## Features

- ✅ **Accurate Thai transcription** using OpenAI Whisper
- ✅ **Speaker diarization** - Automatically separate different speakers
- ✅ **Multiple output formats** - TXT, JSON, SRT (subtitle format)
- ✅ **Timestamps** - Each segment includes timing information
- ✅ **GPU acceleration** - Faster processing with CUDA support
- ✅ **Multiple audio formats** - Supports MP3, WAV, M4A, FLAC, etc.
- ✅ **Fine-tuning** - Fine-tune Whisper, VibeVoice-ASR, or SeamlessM4T on custom Thai audio data
- ✅ **Multi-model support** - Whisper, Microsoft VibeVoice-ASR (9B), Facebook SeamlessM4T v2 (2.3B)
- ✅ **Batch processing** - Transcribe entire directories to CSV with batch scripts

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

**Full version** (with pyannote speaker diarization):
```bash
pip install -r requirements.txt
```

**Simple version** (no pyannote/HuggingFace needed):
```bash
pip install -r requirements_simple.txt
```

**Fine-tuning Whisper** (for training Whisper on custom data):
```bash
pip install -r requirements_finetune.txt
```

**Fine-tuning VibeVoice-ASR** (requires transformers >= 5.3.0):
```bash
pip install -r requirements_finetune.txt
pip install peft
pip install --upgrade transformers>=5.3.0
```

**Fine-tuning SeamlessM4T v2**:
```bash
pip install -r requirements_finetune.txt
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

## Fine-tuning on Custom Thai Data

You can fine-tune multiple ASR models on your own Thai audio dataset. All fine-tuning scripts use the same CSV data format.

### Supported Models for Fine-tuning

| Model | Script | Architecture | Params | Method | License |
|-------|--------|-------------|--------|--------|---------|
| OpenAI Whisper | `finetune_whisper.py` | Encoder-Decoder | 39M–1.5B | Full fine-tune | MIT |
| Microsoft VibeVoice-ASR | `finetune_vibe_voice.py` | Qwen2 LLM + Audio Tokenizers | 9B | LoRA | MIT |
| Facebook SeamlessM4T v2 | `finetune_seamless.py` | UnitY2 Seq2Seq | 2.3B | Full fine-tune | CC-BY-NC-4.0 |

### Data Format

All three fine-tuning scripts use the same data structure:
```
train/
├── annotation/
│   └── train.csv       # CSV with columns: path, sentence
└── audio/
    ├── audio_001.mp3
    ├── audio_002.mp3
    └── ...
val/
├── annotation/
│   └── dev.csv
└── audio/
    └── ...
```

The CSV file format:
```csv
path,sentence
audio_001.mp3,สวัสดีครับ วันนี้เราจะมาประชุม
audio_002.mp3,ขอบคุณครับ ผมมีข้อเสนอ
```

### Loading Models

```python
# --- Whisper (standard) ---
from transformers import WhisperProcessor, WhisperForConditionalGeneration
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

# --- VibeVoice-ASR (9B, custom architecture) ---
from transformers import VibeVoiceForASRTraining
model = VibeVoiceForASRTraining.from_pretrained("microsoft/VibeVoice-ASR", dtype="auto")

# --- SeamlessM4T v2 (2.3B) ---
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("facebook/seamless-m4t-v2-large")
model = AutoModel.from_pretrained("facebook/seamless-m4t-v2-large")
```

### 1. Fine-tuning Whisper

```bash
# Fine-tune with default settings (whisper-base)
python finetune_whisper.py

# Fine-tune with custom settings
python finetune_whisper.py \
    --model_name openai/whisper-medium \
    --batch_size 4 \
    --epochs 5 \
    --learning_rate 1e-5 \
    --output_dir ./whisper-thai-finetuned
```

#### Whisper VRAM Requirements

| Model  | Approximate VRAM |
|--------|-----------------|
| small  | ~8 GB           |
| medium | ~12 GB          |
| large  | ~16 GB+         |

Reduce `--batch_size` if you run out of VRAM. Use `--gradient_accumulation_steps` to maintain effective batch size.

### 2. Fine-tuning VibeVoice-ASR (LoRA)

VibeVoice-ASR is a 9B parameter model — full fine-tuning is impractical, so LoRA is used. The HF-native version (`microsoft/VibeVoice-ASR-HF`) requires `transformers >= 5.3.0`. Falls back to the `vibevoice` package automatically.

```bash
# Default settings
python finetune_vibe_voice.py

# Custom settings
python finetune_vibe_voice.py \
    --model_name microsoft/VibeVoice-ASR-HF \
    --batch_size 1 \
    --epochs 3 \
    --learning_rate 1e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --output_dir ./vibevoice-thai-finetuned
```

#### VibeVoice A100 Recommended Configs

| Config | A100 80GB | A100 40GB | 4x A100 80GB |
|--------|-----------|-----------|-------------|
| `--batch_size` | 2 | 1 | 2 |
| `--gradient_accumulation_steps` | 8 | 16 | 4 |
| `--learning_rate` | 1e-4 | 1e-4 | 1e-4 |
| `--warmup_ratio` | 0.1 | 0.1 | 0.1 |
| `--lora_r` | 16 | 8 | 16 |
| `--lora_alpha` | 32 | 16 | 32 |
| Effective batch size | 16 | 16 | 32 |
| Approx. VRAM usage | ~55 GB | ~35 GB | ~55 GB/GPU |

**A100 80GB (single GPU):**
```bash
python finetune_vibe_voice.py \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.1 \
    --epochs 3 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05
```

**A100 40GB (single GPU):**
```bash
python finetune_vibe_voice.py \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --lora_r 8 \
    --lora_alpha 16
```

**Multi-GPU (4x A100):**
```bash
torchrun --nproc_per_node=4 finetune_vibe_voice.py \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lora_r 16 \
    --lora_alpha 32
```

If you hit OOM, reduce `--batch_size` to 1 and lower `--lora_r` to 8 first.

### 3. Fine-tuning SeamlessM4T v2

SeamlessM4T v2 uses 3-letter language codes (ISO 639-3): `tha` for Thai, `eng` for English.

```bash
# Default settings (Thai)
python finetune_seamless.py

# Custom settings
python finetune_seamless.py \
    --model_name facebook/seamless-m4t-v2-large \
    --batch_size 2 \
    --epochs 3 \
    --learning_rate 1e-5 \
    --tgt_lang tha \
    --output_dir ./seamless-thai-finetuned
```

#### SeamlessM4T v2 VRAM Requirements

| Batch Size | Approximate VRAM |
|-----------|-----------------|
| 1         | ~12 GB          |
| 2         | ~18 GB          |
| 4         | ~28 GB          |

### Inference with Fine-tuned Models

```bash
# --- Whisper ---
python inference_whisper.py --model ./whisper-thai-finetuned --test_dir test --output results.csv
python inference_whisper.py --model ./whisper-thai-finetuned --test_dir test --language th

# --- VibeVoice-ASR (with LoRA adapter) ---
python inference_vibe_voice.py \
    --base_model microsoft/VibeVoice-ASR-HF \
    --lora_path ./vibevoice-thai-finetuned \
    --test_dir test \
    --output results_vibevoice.csv

# --- VibeVoice-ASR (merged model) ---
python inference_vibe_voice.py --model ./vibevoice-merged --test_dir test

# --- SeamlessM4T v2 ---
python inference_fb_seamless.py --model ./seamless-thai-finetuned --test_dir test --output results_seamless.csv
python inference_fb_seamless.py --model ./seamless-thai-finetuned --test_dir test --tgt_lang tha
```

Output CSV format (same for all models):
```csv
path,sentence
LOTUSDIS_000001.mp3,สวัสดีครับ วันนี้เราจะมาประชุม
LOTUSDIS_000002.mp3,ขอบคุณครับ ผมมีข้อเสนอ
LOTUSDIS_000003.mp3,เรามาเริ่มกันเลย
```

You can also use `batch_to_csv.py` for batch transcription:

```bash
# Batch transcribe using the fine-tuned Whisper model
python batch_to_csv.py --finetuned_model ./whisper-thai-finetuned

# With custom input/output paths
python batch_to_csv.py \
    --finetuned_model ./whisper-thai-finetuned \
    --input_dir ./test \
    --output_file submission.csv
```

### Model Comparison for Fine-tuning

| Feature | Whisper | VibeVoice-ASR | SeamlessM4T v2 |
|---------|---------|--------------|----------------|
| Best for | General ASR, well-tested | Long audio (up to 60min), speaker diarization | Multilingual, translation tasks |
| Training method | Full fine-tune | LoRA (parameter-efficient) | Full fine-tune |
| Min GPU | 8 GB (small) | 40 GB (A100) | 12 GB |
| Languages | 99+ | 50+ | 100+ |
| Audio limit | 30 sec chunks | 60 min single pass | Varies |
| License | MIT | MIT | CC-BY-NC-4.0 |

## Batch Transcription to CSV

Transcribe all MP3 files in a directory to a single CSV file:

```bash
# Using openai-whisper (default)
python batch_to_csv.py --input_dir test --output_file submission.csv

# Using a fine-tuned HuggingFace model
python batch_to_csv.py --finetuned_model ./whisper-thai-finetuned
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
├── transcribe_meeting.py          # Full pipeline (Whisper + pyannote diarization)
├── transcribe_meeting_simple.py   # Simple version (pause/clustering-based speakers)
├── simple_transcribe.py           # Minimal standalone example
├── batch_transcribe.py            # Batch process audio directories
├── batch_to_csv.py                # Batch transcribe to CSV (supports fine-tuned models)
├── model_load.py                  # Test loading all supported models
├── finetune_whisper.py            # Fine-tune Whisper on custom data
├── finetune_vibe_voice.py         # Fine-tune VibeVoice-ASR with LoRA
├── finetune_seamless.py           # Fine-tune SeamlessM4T v2
├── inference_whisper.py           # Inference with fine-tuned Whisper (outputs CSV)
├── inference_vibe_voice.py        # Inference with fine-tuned VibeVoice-ASR (outputs CSV)
├── inference_fb_seamless.py       # Inference with fine-tuned SeamlessM4T v2 (outputs CSV)
├── audio_utils.py                 # Audio utilities (convert, normalize, split, info)
├── config.py                      # Configuration constants
├── requirements.txt               # Dependencies (full version)
├── requirements_simple.txt        # Dependencies (simple version)
├── requirements_finetune.txt      # Dependencies (fine-tuning)
├── test_installation.py           # Verify installation
├── README.md                      # This file
└── transcriptions/                # Output directory (created automatically)
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
- **HuggingFace Transformers**: https://github.com/huggingface/transformers
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
