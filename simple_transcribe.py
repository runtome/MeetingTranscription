"""
Simple Example: Basic Thai Audio Transcription
Quick start script without speaker diarization
"""

import whisper

def transcribe_thai_audio(audio_file, model_size="medium"):
    """
    Simple function to transcribe Thai audio
    
    Args:
        audio_file: Path to your audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
    """
    print(f"Loading Whisper {model_size} model...")
    model = whisper.load_model(model_size)
    
    print(f"Transcribing: {audio_file}")
    result = model.transcribe(
        audio_file,
        language="th",  # Thai language
        verbose=True
    )
    
    # Print full transcription
    print("\n" + "="*60)
    print("FULL TRANSCRIPTION:")
    print("="*60)
    print(result["text"])
    
    # Print with timestamps
    print("\n" + "="*60)
    print("TRANSCRIPTION WITH TIMESTAMPS:")
    print("="*60)
    for segment in result["segments"]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        print(f"[{start:.2f}s - {end:.2f}s] {text}")
    
    # Save to file
    output_file = audio_file.rsplit('.', 1)[0] + "_transcript.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result["text"])
    
    print(f"\n✓ Transcription saved to: {output_file}")
    
    return result


if __name__ == "__main__":
    # Example usage
    # Replace with your audio file path
    audio_file = "meeting.mp3"
    
    result = transcribe_thai_audio(audio_file, model_size="medium")
