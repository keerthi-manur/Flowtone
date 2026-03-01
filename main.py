"""
GESTURE.DJ — Main entry point
Drums + Violin, hand gesture controlled
"""

import sys
import os
import threading
import time
import argparse

# Graceful import checks
def check_deps():
    missing = []
    for pkg, imp in [("mediapipe", "mediapipe"), ("opencv-python", "cv2"),
                     ("pygame", "pygame"), ("numpy", "numpy"), ("requests", "requests")]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("❌ Missing packages. Run:\n")
        print(f"  pip install {' '.join(missing)}\n")
        sys.exit(1)

check_deps()

from gesture_engine import GestureEngine
from sound_engine import SoundEngine
from elevenlabs_client import ElevenLabsClient
from overlay import Overlay

def main():
    parser = argparse.ArgumentParser(description="Gesture.DJ — Play instruments with your hands")
    parser.add_argument("--elevenlabs-key", default=os.getenv("ELEVENLABS_API_KEY", ""),
                        help="ElevenLabs API key (or set ELEVENLABS_API_KEY env var)")
    parser.add_argument("--samples-dir", default="samples",
                        help="Directory containing audio samples")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default 0)")
    parser.add_argument("--no-voice", action="store_true",
                        help="Disable ElevenLabs voice announcements")
    args = parser.parse_args()

    print("🥁 GESTURE.DJ starting up...\n")

    # Init sound engine first (pre-loads all samples)
    print("🔊 Loading audio samples...")
    sound = SoundEngine(args.samples_dir)
    sound.preload_all()
    print(f"   ✓ {sound.sample_count()} samples loaded\n")

    # Init ElevenLabs (optional)
    voice = None
    if args.elevenlabs_key and not args.no_voice:
        print("🎙  Connecting to ElevenLabs...")
        voice = ElevenLabsClient(args.elevenlabs_key)
        # Pre-generate startup phrases so there's no latency mid-performance
        voice.prewarm([
            "Drum mode",
            "Violin mode",
            "Switching to drums",
            "Switching to violin",
            "Muted",
            "Reverb on",
            "Reverb off",
        ])
        print("   ✓ Voice ready\n")
    else:
        print("   ℹ️  No ElevenLabs key — voice disabled (pass --elevenlabs-key or set ELEVENLABS_API_KEY)\n")

    # Init overlay (the OpenCV window)
    overlay = Overlay()

    # Init gesture engine (MediaPipe)
    print("👋 Starting gesture detection...")
    engine = GestureEngine(
        camera_index=args.camera,
        sound=sound,
        voice=voice,
        overlay=overlay,
    )
    print("   ✓ Camera open\n")
    print("━" * 50)
    print("  CONTROLS")
    print("━" * 50)
    print("  LEFT HAND  — pitch (height) + instrument mode")
    print("  RIGHT HAND — volume (height) + effects")
    print()
    print("  DRUM MODE gestures:")
    print("    ✊ Fist         → Kick drum")
    print("    ☝ 1 finger     → Snare")
    print("    ✌ 2 fingers    → Hi-hat")
    print("    🤟 3 fingers    → Tom")
    print("    🖐 Open palm    → Cymbal crash")
    print()
    print("  VIOLIN MODE gestures:")
    print("    Left hand ↕    → Pitch (pentatonic scale)")
    print("    Right hand ↕   → Bow pressure / volume")
    print("    👋 Wave         → Vibrato")
    print("    ✊ Fist         → Mute")
    print()
    print("  BOTH MODES:")
    print("    👍 Thumbs up    → Switch instrument")
    print("    Q               → Quit")
    print("━" * 50)
    print()

    if voice:
        # Announce startup in background so it doesn't block
        threading.Thread(target=voice.speak, args=("Drum mode",), daemon=True).start()

    engine.run()  # blocking loop — exits when user presses Q

    print("\n👋 Goodbye!")
    sound.shutdown()


if __name__ == "__main__":
    main()
