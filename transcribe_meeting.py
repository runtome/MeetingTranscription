"""
Thai Meeting Transcription Tool
Uses OpenAI Whisper for speech recognition and pyannote.audio for speaker diarization
"""

import os
import argparse
import json
from datetime import timedelta
from pathlib import Path
import torch
import whisper
from pyannote.audio import Pipeline
import warnings
warnings.filterwarnings("ignore")


class MeetingTranscriber:
    def __init__(self, whisper_model="medium", language="th", device=None):
        """
        Initialize the transcriber with Whisper model and diarization pipeline
        
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            language: Language code (default: "th" for Thai)
            device: torch device (cuda/cpu), auto-detected if None
        """
        self.language = language
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load Whisper model
        print(f"Loading Whisper {whisper_model} model...")
        self.whisper_model = whisper.load_model(whisper_model, device=self.device)
        
        # Initialize diarization pipeline (requires HuggingFace token)
        self.diarization_pipeline = None
        
    def load_diarization(self, hf_token):
        """
        Load speaker diarization pipeline
        
        Args:
            hf_token: HuggingFace access token (required for pyannote models)
        """
        print("Loading speaker diarization pipeline...")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        if self.device == "cuda":
            self.diarization_pipeline.to(torch.device("cuda"))
    
    def transcribe_audio(self, audio_path, verbose=True):
        """
        Transcribe audio file using Whisper
        
        Args:
            audio_path: Path to audio file
            verbose: Print progress
            
        Returns:
            Whisper transcription result
        """
        print(f"\nTranscribing: {audio_path}")
        
        result = self.whisper_model.transcribe(
            audio_path,
            language=self.language,
            verbose=verbose,
            word_timestamps=True  # Enable word-level timestamps
        )
        
        return result
    
    def diarize_audio(self, audio_path):
        """
        Perform speaker diarization on audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Diarization result with speaker segments
        """
        if self.diarization_pipeline is None:
            raise ValueError("Diarization pipeline not loaded. Call load_diarization() first.")
        
        print("Performing speaker diarization...")
        diarization = self.diarization_pipeline(audio_path)
        
        return diarization
    
    def merge_transcription_with_speakers(self, transcription, diarization):
        """
        Merge Whisper transcription with speaker diarization
        
        Args:
            transcription: Whisper transcription result
            diarization: pyannote diarization result
            
        Returns:
            List of segments with speaker labels and text
        """
        # Extract speaker timeline
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
        
        # Merge with transcription segments
        merged_segments = []
        
        for segment in transcription['segments']:
            seg_start = segment['start']
            seg_end = segment['end']
            seg_text = segment['text'].strip()
            
            # Find overlapping speaker
            speaker = "Unknown"
            max_overlap = 0
            
            for spk_seg in speaker_segments:
                # Calculate overlap
                overlap_start = max(seg_start, spk_seg['start'])
                overlap_end = min(seg_end, spk_seg['end'])
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    speaker = spk_seg['speaker']
            
            merged_segments.append({
                'start': seg_start,
                'end': seg_end,
                'speaker': speaker,
                'text': seg_text
            })
        
        return merged_segments
    
    def format_time(self, seconds):
        """Convert seconds to HH:MM:SS format"""
        return str(timedelta(seconds=int(seconds)))
    
    def save_transcription(self, segments, output_path, format_type='txt'):
        """
        Save transcription to file
        
        Args:
            segments: List of transcription segments
            output_path: Output file path
            format_type: Output format ('txt', 'json', 'srt')
        """
        output_path = Path(output_path)
        
        if format_type == 'txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                current_speaker = None
                for seg in segments:
                    # Add speaker label when speaker changes
                    if seg['speaker'] != current_speaker:
                        current_speaker = seg['speaker']
                        f.write(f"\n[{current_speaker}]\n")
                    
                    timestamp = self.format_time(seg['start'])
                    f.write(f"[{timestamp}] {seg['text']}\n")
        
        elif format_type == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)
        
        elif format_type == 'srt':
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, seg in enumerate(segments, 1):
                    start_time = self.format_timestamp_srt(seg['start'])
                    end_time = self.format_timestamp_srt(seg['end'])
                    speaker = seg['speaker']
                    text = seg['text']
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"[{speaker}] {text}\n\n")
        
        print(f"\nTranscription saved to: {output_path}")
    
    def format_timestamp_srt(self, seconds):
        """Format timestamp for SRT subtitle format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def process_meeting(self, audio_path, output_dir, with_speakers=True, hf_token=None):
        """
        Complete pipeline: transcribe and optionally add speaker labels
        
        Args:
            audio_path: Path to audio file
            output_dir: Directory to save outputs
            with_speakers: Whether to perform speaker diarization
            hf_token: HuggingFace token (required if with_speakers=True)
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Transcribe
        transcription = self.transcribe_audio(str(audio_path))
        
        # Perform speaker diarization if requested
        if with_speakers:
            if hf_token is None:
                raise ValueError("HuggingFace token required for speaker diarization")
            
            if self.diarization_pipeline is None:
                self.load_diarization(hf_token)
            
            diarization = self.diarize_audio(str(audio_path))
            segments = self.merge_transcription_with_speakers(transcription, diarization)
        else:
            # No speaker labels
            segments = [
                {
                    'start': seg['start'],
                    'end': seg['end'],
                    'speaker': 'Speaker',
                    'text': seg['text'].strip()
                }
                for seg in transcription['segments']
            ]
        
        # Save in multiple formats
        base_name = audio_path.stem
        
        self.save_transcription(
            segments,
            output_dir / f"{base_name}_transcript.txt",
            format_type='txt'
        )
        
        self.save_transcription(
            segments,
            output_dir / f"{base_name}_transcript.json",
            format_type='json'
        )
        
        self.save_transcription(
            segments,
            output_dir / f"{base_name}_transcript.srt",
            format_type='srt'
        )
        
        print("\n" + "="*60)
        print("TRANSCRIPTION COMPLETE")
        print("="*60)
        print(f"Audio file: {audio_path.name}")
        print(f"Output directory: {output_dir}")
        print(f"Speaker diarization: {'Enabled' if with_speakers else 'Disabled'}")
        print(f"Number of segments: {len(segments)}")
        if with_speakers:
            speakers = set(seg['speaker'] for seg in segments)
            print(f"Number of speakers detected: {len(speakers)}")
            print(f"Speakers: {', '.join(sorted(speakers))}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Transcribe Thai meeting audio with speaker diarization'
    )
    
    parser.add_argument(
        'audio_file',
        type=str,
        help='Path to audio file (mp3, wav, m4a, etc.)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='./transcriptions',
        help='Output directory for transcriptions (default: ./transcriptions)'
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='medium',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model size (default: medium)'
    )
    
    parser.add_argument(
        '-l', '--language',
        type=str,
        default='th',
        help='Language code (default: th for Thai)'
    )
    
    parser.add_argument(
        '--no-speakers',
        action='store_true',
        help='Disable speaker diarization (faster but no speaker labels)'
    )
    
    parser.add_argument(
        '--hf-token',
        type=str,
        default=None,
        help='HuggingFace access token (required for speaker diarization)'
    )
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        return
    
    # Initialize transcriber
    transcriber = MeetingTranscriber(
        whisper_model=args.model,
        language=args.language
    )
    
    # Process meeting
    try:
        transcriber.process_meeting(
            audio_path=args.audio_file,
            output_dir=args.output,
            with_speakers=not args.no_speakers,
            hf_token=args.hf_token
        )
    except Exception as e:
        print(f"\nError during processing: {e}")
        print("\nIf using speaker diarization, make sure you:")
        print("1. Have a HuggingFace account and token")
        print("2. Accepted the pyannote model terms at:")
        print("   https://huggingface.co/pyannote/speaker-diarization-3.1")
        raise


if __name__ == "__main__":
    main()
