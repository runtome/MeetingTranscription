"""
Thai Meeting Transcription Tool (Alternative Version)
Uses OpenAI Whisper for speech recognition with simple speaker detection
No pyannote dependency - uses voice activity detection and clustering
"""

import os
import argparse
import json
from datetime import timedelta
from pathlib import Path
import torch
import whisper
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import warnings
warnings.filterwarnings("ignore")


class SimpleMeetingTranscriber:
    def __init__(self, whisper_model="medium", language="th", device=None):
        """
        Initialize the transcriber with Whisper model
        
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
    
    def simple_speaker_detection(self, audio_path, transcription, num_speakers=None):
        """
        Simple speaker detection using audio features and clustering
        This is a basic implementation that doesn't require pyannote
        
        Args:
            audio_path: Path to audio file
            transcription: Whisper transcription result
            num_speakers: Expected number of speakers (auto-detect if None)
            
        Returns:
            List of segments with speaker labels
        """
        print("Performing simple speaker detection...")
        
        # Load audio
        audio = whisper.load_audio(audio_path)
        
        segments_with_speakers = []
        
        # Extract features for each segment
        features_list = []
        valid_segments = []
        
        for segment in transcription['segments']:
            start_sample = int(segment['start'] * 16000)
            end_sample = int(segment['end'] * 16000)
            
            # Extract segment audio
            segment_audio = audio[start_sample:end_sample]
            
            if len(segment_audio) > 0:
                # Simple feature extraction: spectral statistics
                # This is a basic approach - pyannote uses more sophisticated methods
                features = self._extract_simple_features(segment_audio)
                features_list.append(features)
                valid_segments.append(segment)
        
        if len(features_list) == 0:
            print("Warning: No valid segments for speaker detection")
            return self._add_default_speakers(transcription['segments'])
        
        # Convert to numpy array
        features_array = np.array(features_list)
        
        # Determine number of speakers
        if num_speakers is None:
            # Auto-detect (simple heuristic: 2-4 speakers)
            num_speakers = min(4, max(2, len(valid_segments) // 10))
        
        print(f"Detecting {num_speakers} speakers...")
        
        # Cluster segments by speaker
        if len(features_array) < num_speakers:
            # Not enough segments, just assign sequential speakers
            speaker_labels = list(range(len(features_array)))
        else:
            clustering = AgglomerativeClustering(
                n_clusters=num_speakers,
                linkage='ward'
            )
            speaker_labels = clustering.fit_predict(features_array)
        
        # Assign speakers to segments
        for i, segment in enumerate(valid_segments):
            speaker_id = speaker_labels[i]
            segments_with_speakers.append({
                'start': segment['start'],
                'end': segment['end'],
                'speaker': f'SPEAKER_{speaker_id:02d}',
                'text': segment['text'].strip()
            })
        
        return segments_with_speakers
    
    def _extract_simple_features(self, audio_segment):
        """
        Extract simple audio features for speaker detection
        This is a basic implementation - not as accurate as pyannote
        
        Args:
            audio_segment: Audio samples
            
        Returns:
            Feature vector
        """
        # Ensure we have enough samples
        if len(audio_segment) < 100:
            return np.zeros(10)
        
        # Simple features:
        # 1. Energy statistics
        energy = np.abs(audio_segment)
        mean_energy = np.mean(energy)
        std_energy = np.std(energy)
        
        # 2. Zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_segment)))) / (2 * len(audio_segment))
        
        # 3. Spectral features (basic FFT)
        fft = np.fft.fft(audio_segment)
        magnitude = np.abs(fft[:len(fft)//2])
        
        # Spectral centroid
        freqs = np.fft.fftfreq(len(audio_segment), 1/16000)[:len(fft)//2]
        spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
        
        # Spectral rolloff
        cumsum = np.cumsum(magnitude)
        rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
        spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        
        # 4. Pitch estimation (very basic)
        autocorr = np.correlate(audio_segment, audio_segment, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find first peak after initial
        peaks = []
        for i in range(20, min(400, len(autocorr)-1)):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                peaks.append((i, autocorr[i]))
        
        if peaks:
            pitch_period = max(peaks, key=lambda x: x[1])[0]
            pitch = 16000 / pitch_period
        else:
            pitch = 0
        
        # Combine features
        features = np.array([
            mean_energy,
            std_energy,
            zero_crossings,
            spectral_centroid,
            spectral_rolloff,
            pitch,
            np.mean(magnitude[:10]),  # Low frequency energy
            np.mean(magnitude[10:50]),  # Mid frequency energy
            np.mean(magnitude[50:100]),  # High frequency energy
            np.percentile(magnitude, 75)  # Spectral spread
        ])
        
        return features
    
    def _add_default_speakers(self, segments):
        """Add default speaker labels when detection fails"""
        return [
            {
                'start': seg['start'],
                'end': seg['end'],
                'speaker': 'SPEAKER_00',
                'text': seg['text'].strip()
            }
            for seg in segments
        ]
    
    def pause_based_speaker_detection(self, transcription, pause_threshold=2.0):
        """
        Alternative: Detect speaker changes based on pauses
        Assumes speaker changes happen after long pauses
        
        Args:
            transcription: Whisper transcription result
            pause_threshold: Minimum pause duration (seconds) to trigger speaker change
            
        Returns:
            List of segments with speaker labels
        """
        print(f"Using pause-based speaker detection (threshold: {pause_threshold}s)...")
        
        segments_with_speakers = []
        current_speaker = 0
        
        for i, segment in enumerate(transcription['segments']):
            # Check pause before this segment
            if i > 0:
                previous_end = transcription['segments'][i-1]['end']
                current_start = segment['start']
                pause_duration = current_start - previous_end
                
                # If long pause, assume speaker change
                if pause_duration > pause_threshold:
                    current_speaker += 1
            
            segments_with_speakers.append({
                'start': segment['start'],
                'end': segment['end'],
                'speaker': f'SPEAKER_{current_speaker:02d}',
                'text': segment['text'].strip()
            })
        
        # Count unique speakers
        unique_speakers = len(set(seg['speaker'] for seg in segments_with_speakers))
        print(f"Detected {unique_speakers} speakers based on pauses")
        
        return segments_with_speakers
    
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
    
    def process_meeting(self, audio_path, output_dir, detection_method='pause', 
                       num_speakers=None, pause_threshold=2.0):
        """
        Complete pipeline: transcribe and add speaker labels
        
        Args:
            audio_path: Path to audio file
            output_dir: Directory to save outputs
            detection_method: 'pause' or 'clustering'
            num_speakers: Number of speakers (for clustering method)
            pause_threshold: Pause threshold in seconds (for pause method)
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Transcribe
        transcription = self.transcribe_audio(str(audio_path))
        
        # Add speaker labels
        if detection_method == 'clustering':
            segments = self.simple_speaker_detection(
                str(audio_path), 
                transcription, 
                num_speakers=num_speakers
            )
        elif detection_method == 'pause':
            segments = self.pause_based_speaker_detection(
                transcription, 
                pause_threshold=pause_threshold
            )
        else:
            # No speaker detection
            segments = [
                {
                    'start': seg['start'],
                    'end': seg['end'],
                    'speaker': 'SPEAKER_00',
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
        print(f"Detection method: {detection_method}")
        print(f"Number of segments: {len(segments)}")
        speakers = set(seg['speaker'] for seg in segments)
        print(f"Number of speakers detected: {len(speakers)}")
        print(f"Speakers: {', '.join(sorted(speakers))}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Transcribe Thai meeting audio with simple speaker detection'
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
        '--method',
        type=str,
        default='pause',
        choices=['pause', 'clustering', 'none'],
        help='Speaker detection method: pause (default), clustering, or none'
    )
    
    parser.add_argument(
        '--speakers',
        type=int,
        default=None,
        help='Number of speakers (for clustering method)'
    )
    
    parser.add_argument(
        '--pause-threshold',
        type=float,
        default=2.0,
        help='Pause threshold in seconds (default: 2.0)'
    )
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        return
    
    # Initialize transcriber
    transcriber = SimpleMeetingTranscriber(
        whisper_model=args.model,
        language=args.language
    )
    
    # Process meeting
    try:
        transcriber.process_meeting(
            audio_path=args.audio_file,
            output_dir=args.output,
            detection_method=args.method,
            num_speakers=args.speakers,
            pause_threshold=args.pause_threshold
        )
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()