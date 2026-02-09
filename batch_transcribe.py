"""
Batch Transcription Script
Process multiple audio files in a directory
"""

import os
import argparse
from pathlib import Path
from transcribe_meeting import MeetingTranscriber


def batch_transcribe(input_dir, output_dir, model_size="medium", 
                     with_speakers=True, hf_token=None):
    """
    Transcribe all audio files in a directory
    
    Args:
        input_dir: Directory containing audio files
        output_dir: Directory to save transcriptions
        model_size: Whisper model size
        with_speakers: Enable speaker diarization
        hf_token: HuggingFace token
    """
    # Supported audio extensions
    audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac'}
    
    # Find all audio files
    input_path = Path(input_dir)
    audio_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in audio_extensions
    ]
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files to process")
    print("="*60)
    
    # Initialize transcriber once for all files
    transcriber = MeetingTranscriber(
        whisper_model=model_size,
        language="th"
    )
    
    # Load diarization pipeline if needed
    if with_speakers and hf_token:
        transcriber.load_diarization(hf_token)
    
    # Process each file
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\nProcessing file {i}/{len(audio_files)}: {audio_file.name}")
        print("-"*60)
        
        try:
            # Create output subdirectory for this file
            file_output_dir = Path(output_dir) / audio_file.stem
            
            transcriber.process_meeting(
                audio_path=str(audio_file),
                output_dir=str(file_output_dir),
                with_speakers=with_speakers,
                hf_token=hf_token
            )
            
            print(f"✓ Successfully processed: {audio_file.name}")
            
        except Exception as e:
            print(f"✗ Error processing {audio_file.name}: {e}")
            continue
    
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)
    print(f"Total files processed: {len(audio_files)}")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Batch transcribe multiple Thai audio files'
    )
    
    parser.add_argument(
        'input_dir',
        type=str,
        help='Directory containing audio files'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='./batch_transcriptions',
        help='Output directory (default: ./batch_transcriptions)'
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='medium',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model size (default: medium)'
    )
    
    parser.add_argument(
        '--no-speakers',
        action='store_true',
        help='Disable speaker diarization'
    )
    
    parser.add_argument(
        '--hf-token',
        type=str,
        default=None,
        help='HuggingFace access token'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return
    
    # Process all files
    batch_transcribe(
        input_dir=args.input_dir,
        output_dir=args.output,
        model_size=args.model,
        with_speakers=not args.no_speakers,
        hf_token=args.hf_token
    )


if __name__ == "__main__":
    main()
