"""
Audio Preprocessing Utilities
Convert, normalize, and prepare audio files for transcription
"""

import os
import argparse
from pathlib import Path
from pydub import AudioSegment
from pydub.effects import normalize


class AudioPreprocessor:
    """Utility class for audio preprocessing"""
    
    @staticmethod
    def convert_to_wav(input_file, output_file=None, sample_rate=16000):
        """
        Convert any audio format to WAV
        
        Args:
            input_file: Input audio file path
            output_file: Output WAV file path (optional)
            sample_rate: Target sample rate in Hz (default: 16000)
        """
        input_path = Path(input_file)
        
        if output_file is None:
            output_file = input_path.with_suffix('.wav')
        
        print(f"Converting {input_file} to WAV...")
        
        # Load audio
        audio = AudioSegment.from_file(input_file)
        
        # Set sample rate
        audio = audio.set_frame_rate(sample_rate)
        
        # Convert to mono if stereo
        if audio.channels > 1:
            print("Converting to mono...")
            audio = audio.set_channels(1)
        
        # Export as WAV
        audio.export(output_file, format='wav')
        
        print(f"✓ Saved to: {output_file}")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Channels: {audio.channels}")
        print(f"  Duration: {len(audio) / 1000:.2f} seconds")
        
        return output_file
    
    @staticmethod
    def normalize_audio(input_file, output_file=None):
        """
        Normalize audio volume
        
        Args:
            input_file: Input audio file path
            output_file: Output file path (optional)
        """
        input_path = Path(input_file)
        
        if output_file is None:
            output_file = input_path.parent / f"{input_path.stem}_normalized{input_path.suffix}"
        
        print(f"Normalizing {input_file}...")
        
        # Load and normalize
        audio = AudioSegment.from_file(input_file)
        normalized_audio = normalize(audio)
        
        # Export
        normalized_audio.export(output_file, format=input_path.suffix[1:])
        
        print(f"✓ Normalized audio saved to: {output_file}")
        
        return output_file
    
    @staticmethod
    def split_audio(input_file, output_dir, segment_length_ms=600000):
        """
        Split long audio into smaller segments
        
        Args:
            input_file: Input audio file path
            output_dir: Directory to save segments
            segment_length_ms: Segment length in milliseconds (default: 10 minutes)
        """
        input_path = Path(input_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading {input_file}...")
        audio = AudioSegment.from_file(input_file)
        
        total_length = len(audio)
        segment_count = (total_length + segment_length_ms - 1) // segment_length_ms
        
        print(f"Splitting into {segment_count} segments of {segment_length_ms/1000/60:.1f} minutes...")
        
        segments = []
        for i in range(segment_count):
            start = i * segment_length_ms
            end = min((i + 1) * segment_length_ms, total_length)
            
            segment = audio[start:end]
            
            output_file = output_path / f"{input_path.stem}_part{i+1:02d}{input_path.suffix}"
            segment.export(output_file, format=input_path.suffix[1:])
            
            segments.append(output_file)
            print(f"  ✓ Segment {i+1}/{segment_count}: {output_file.name}")
        
        print(f"\n✓ Split complete! {len(segments)} segments saved to {output_dir}")
        
        return segments
    
    @staticmethod
    def get_audio_info(input_file):
        """
        Get detailed information about an audio file
        
        Args:
            input_file: Audio file path
        """
        print(f"Audio file information: {input_file}")
        print("="*60)
        
        audio = AudioSegment.from_file(input_file)
        
        duration_seconds = len(audio) / 1000
        duration_minutes = duration_seconds / 60
        
        print(f"Duration:     {duration_minutes:.2f} minutes ({duration_seconds:.2f} seconds)")
        print(f"Sample rate:  {audio.frame_rate} Hz")
        print(f"Channels:     {audio.channels} ({'stereo' if audio.channels == 2 else 'mono'})")
        print(f"Bit depth:    {audio.sample_width * 8} bit")
        print(f"Format:       {Path(input_file).suffix[1:].upper()}")
        
        file_size = os.path.getsize(input_file)
        print(f"File size:    {file_size / 1024 / 1024:.2f} MB")
        
        return {
            'duration_seconds': duration_seconds,
            'sample_rate': audio.frame_rate,
            'channels': audio.channels,
            'bit_depth': audio.sample_width * 8,
            'file_size_mb': file_size / 1024 / 1024
        }


def main():
    parser = argparse.ArgumentParser(
        description='Audio preprocessing utilities for Thai transcription'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert audio to WAV')
    convert_parser.add_argument('input', type=str, help='Input audio file')
    convert_parser.add_argument('-o', '--output', type=str, help='Output WAV file')
    convert_parser.add_argument('-r', '--rate', type=int, default=16000, help='Sample rate (default: 16000)')
    
    # Normalize command
    normalize_parser = subparsers.add_parser('normalize', help='Normalize audio volume')
    normalize_parser.add_argument('input', type=str, help='Input audio file')
    normalize_parser.add_argument('-o', '--output', type=str, help='Output file')
    
    # Split command
    split_parser = subparsers.add_parser('split', help='Split long audio into segments')
    split_parser.add_argument('input', type=str, help='Input audio file')
    split_parser.add_argument('-o', '--output', type=str, default='./segments', help='Output directory')
    split_parser.add_argument('-l', '--length', type=int, default=10, help='Segment length in minutes (default: 10)')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show audio file information')
    info_parser.add_argument('input', type=str, help='Input audio file')
    
    args = parser.parse_args()
    
    preprocessor = AudioPreprocessor()
    
    if args.command == 'convert':
        preprocessor.convert_to_wav(args.input, args.output, args.rate)
    
    elif args.command == 'normalize':
        preprocessor.normalize_audio(args.input, args.output)
    
    elif args.command == 'split':
        segment_length_ms = args.length * 60 * 1000
        preprocessor.split_audio(args.input, args.output, segment_length_ms)
    
    elif args.command == 'info':
        preprocessor.get_audio_info(args.input)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
