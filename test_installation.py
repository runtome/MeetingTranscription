"""
Test Installation Script
Verify that all dependencies are correctly installed
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    print("="*60)
    
    tests = {
        "torch": "PyTorch",
        "whisper": "OpenAI Whisper",
        "pyannote.audio": "Pyannote Audio",
        "pyannote.core": "Pyannote Core",
        "pydub": "Pydub",
        "numpy": "NumPy",
    }
    
    results = {}
    
    for module, name in tests.items():
        try:
            if module == "pyannote.audio":
                from pyannote.audio import Pipeline
            elif module == "pyannote.core":
                from pyannote.core import Segment
            else:
                __import__(module)
            
            print(f"✓ {name:20s} - OK")
            results[name] = True
        except ImportError as e:
            print(f"✗ {name:20s} - FAILED ({e})")
            results[name] = False
    
    return results


def test_cuda():
    """Test CUDA availability"""
    print("\n" + "="*60)
    print("Testing CUDA/GPU Support...")
    print("="*60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU device: {torch.cuda.get_device_name(0)}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("⚠ CUDA not available - will use CPU (slower)")
            print("  This is OK, but processing will be slower")
            return False
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False


def test_ffmpeg():
    """Test FFmpeg installation"""
    print("\n" + "="*60)
    print("Testing FFmpeg...")
    print("="*60)
    
    try:
        import subprocess
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✓ FFmpeg installed: {version_line}")
            return True
        else:
            print("✗ FFmpeg not working properly")
            return False
    except FileNotFoundError:
        print("✗ FFmpeg not found")
        print("  Install: sudo apt install ffmpeg  (Ubuntu)")
        print("          brew install ffmpeg       (macOS)")
        return False
    except Exception as e:
        print(f"✗ Error checking FFmpeg: {e}")
        return False


def test_whisper():
    """Test Whisper model loading"""
    print("\n" + "="*60)
    print("Testing Whisper Model Loading...")
    print("="*60)
    
    try:
        import whisper
        print("Loading tiny model (fastest, for testing only)...")
        model = whisper.load_model("tiny")
        print("✓ Whisper model loaded successfully")
        print("  You can now use larger models (base, small, medium, large)")
        return True
    except Exception as e:
        print(f"✗ Error loading Whisper model: {e}")
        return False


def main():
    print("\n")
    print("="*60)
    print("THAI MEETING TRANSCRIPTION - INSTALLATION TEST")
    print("="*60)
    print()
    
    # Test all components
    import_results = test_imports()
    cuda_ok = test_cuda()
    ffmpeg_ok = test_ffmpeg()
    whisper_ok = test_whisper()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_imports_ok = all(import_results.values())
    
    if all_imports_ok and ffmpeg_ok and whisper_ok:
        print("✓ All tests passed!")
        print("\nYou're ready to transcribe Thai meetings!")
        print("\nQuick start:")
        print("  python simple_transcribe.py")
        print("  python transcribe_meeting.py meeting.mp3 --no-speakers")
        
        if not cuda_ok:
            print("\nNote: CUDA not available - processing will use CPU")
            print("      This is slower but still works fine.")
    else:
        print("⚠ Some tests failed. Please fix the issues above.")
        
        if not all_imports_ok:
            print("\nMissing packages. Try:")
            print("  pip install -r requirements.txt")
        
        if not ffmpeg_ok:
            print("\nFFmpeg missing. Install it:")
            print("  Ubuntu: sudo apt install ffmpeg")
            print("  macOS:  brew install ffmpeg")
    
    print("="*60)


if __name__ == "__main__":
    main()
