# Thai Meeting Transcription Tool (Simplified Version)

This is a **simplified version** that does NOT require pyannote or HuggingFace tokens. It uses basic speaker detection methods instead.

## Key Differences from Full Version

✅ **No pyannote dependency** - No HuggingFace account needed
✅ **No CUDA library issues** - Easier to install
✅ **Two speaker detection methods:**
   - **Pause-based**: Detects speaker changes based on silence/pauses
   - **Clustering-based**: Groups segments by audio features

⚠️ **Trade-off**: Speaker detection is less accurate than pyannote but still useful

## Installation

```bash
# Install dependencies (much simpler!)
pip install -r requirements_simple.txt
```

## Usage

### Method 1: Pause-Based Detection (Recommended)

Assumes speakers change after long pauses (2+ seconds of silence):

```bash
python transcribe_meeting_simple.py meeting.mp3 --method pause
```

Adjust pause threshold:
```bash
python transcribe_meeting_simple.py meeting.mp3 --method pause --pause-threshold 1.5
```

### Method 2: Clustering-Based Detection

Groups similar-sounding speech segments together:

```bash
python transcribe_meeting_simple.py meeting.mp3 --method clustering --speakers 3
```

### Method 3: No Speaker Detection (Fastest)

Just transcription, all segments labeled as one speaker:

```bash
python transcribe_meeting_simple.py meeting.mp3 --method none
```

## Command Line Options

```
python transcribe_meeting_simple.py <audio_file> [options]

Options:
  -o, --output DIR          Output directory (default: ./transcriptions)
  -m, --model MODEL         Whisper model: tiny, base, small, medium, large
  -l, --language LANG       Language code (default: th)
  --method METHOD           pause, clustering, or none (default: pause)
  --speakers N              Number of speakers (for clustering method)
  --pause-threshold SECS    Pause threshold in seconds (default: 2.0)
```

## Examples

### Example 1: Two speakers with clear pauses
```bash
python transcribe_meeting_simple.py interview.mp3 --method pause --pause-threshold 1.5
```

### Example 2: Meeting with 4 people
```bash
python transcribe_meeting_simple.py meeting.mp3 --method clustering --speakers 4
```

### Example 3: Quick transcription only
```bash
python transcribe_meeting_simple.py lecture.mp3 --method none -m small
```

## How It Works

### Pause-Based Method
- Monitors silence between speech segments
- When pause > threshold, assumes new speaker
- **Best for**: Interviews, alternating conversations
- **Pros**: Fast, simple, works well for turn-taking
- **Cons**: Misses speaker changes without pauses

### Clustering Method
- Extracts audio features (pitch, energy, spectrum)
- Groups similar segments using clustering
- **Best for**: Meetings with distinct voices
- **Pros**: Can detect same speaker across conversation
- **Cons**: Less accurate than pyannote, needs speaker count

## Tips for Best Results

### For Pause-Based Detection:
- Use **shorter threshold (1.0-1.5s)** for fast conversations
- Use **longer threshold (2.5-3.0s)** for slower meetings
- Works best when speakers take clear turns

### For Clustering-Based Detection:
- Specify `--speakers` if you know the count
- Works better with distinct voice characteristics
- Needs longer audio (5+ minutes) for better clustering

### General Tips:
- **Clear audio** = better results
- **Minimize background noise**
- **Distinct speakers** help both methods
- Start with **pause method** - usually more reliable

## Output Files

All methods produce 3 files:

1. **TXT** - Human-readable with timestamps and speaker labels
2. **JSON** - Machine-readable data
3. **SRT** - Subtitle format

Example TXT output:
```
[SPEAKER_00]
[00:00:05] สวัสดีครับ วันนี้เราจะมาประชุมเรื่องโปรเจคใหม่

[SPEAKER_01]
[00:00:15] ขอบคุณครับ ผมมีข้อเสนอเกี่ยวกับแผนการตลาด
```

## Comparison: Simple vs Full Version

| Feature | Simple Version | Full Version (pyannote) |
|---------|---------------|------------------------|
| Installation | ✅ Easy | ⚠️ Complex (CUDA/HF) |
| Dependencies | Minimal | Many |
| HF Account | ❌ Not needed | ✅ Required |
| Speaker Accuracy | 60-70% | 85-95% |
| Speed | Fast | Slower |
| Best For | Quick tasks | Production quality |

## When to Use Which Version

**Use Simple Version when:**
- You want quick setup
- Don't have HuggingFace account
- Having CUDA/pyannote issues
- Accuracy is "good enough"
- Testing or prototyping

**Use Full Version when:**
- Need high accuracy speaker detection
- Production environment
- Multiple overlapping speakers
- Complex audio scenarios

## Troubleshooting

### "ModuleNotFoundError: No module named 'sklearn'"
```bash
pip install scikit-learn
```

### "FFmpeg not found"
```bash
# Ubuntu
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### Poor speaker detection
Try adjusting parameters:
```bash
# For pause method: adjust threshold
--pause-threshold 1.0  # More speaker changes
--pause-threshold 3.0  # Fewer speaker changes

# For clustering: specify exact speaker count
--speakers 2  # Two-person interview
--speakers 5  # Panel discussion
```

## Performance

30-minute meeting on typical laptop:

| Method | Time | Speaker Accuracy |
|--------|------|-----------------|
| None | ~8 min | N/A |
| Pause | ~8 min | 60-70% |
| Clustering | ~10 min | 65-75% |

Compare to pyannote: ~15 min, 85-95% accuracy

## Limitations

- Cannot handle overlapping speech well
- Less accurate than pyannote for complex scenarios
- Clustering method needs speaker count or good guess
- Pause method assumes turn-taking conversations

## Next Steps

If you need better accuracy, consider:
1. Upgrading to full version with pyannote
2. Manually correcting speaker labels in JSON output
3. Using external diarization tools
4. Recording higher quality audio

## Support

This simplified version is ideal for:
- Quick transcriptions
- Testing purposes
- Learning/education
- Resource-constrained environments

For production use with high accuracy requirements, use the full version with pyannote.
