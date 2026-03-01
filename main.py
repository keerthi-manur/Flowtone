"""
GESTURE.DJ — Main entry point
Drums + Violin, hand gesture controlled via webcam.
"""

import sys
import os
import threading
import argparse


def check_deps():
    missing = []
    for pkg, imp in [("mediapipe","mediapipe"),("opencv-python","cv2"),
                     ("pygame","pygame"),("numpy","numpy"),("requests","requests")]:
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
from sound_engine   import SoundEngine
from elevenlabs_client import ElevenLabsClient
from overlay        import Overlay


def main():
    parser = argparse.ArgumentParser(description="Gesture.DJ")
    parser.add_argument("--elevenlabs-key", default=os.getenv("ELEVENLABS_API_KEY",""),
                        help="ElevenLabs API key (or set ELEVENLABS_API_KEY env var)")
    parser.add_argument("--samples-dir", default="samples")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index. Try 1 or 2 if 0 shows wrong camera.")
    parser.add_argument("--mirror", action="store_true",
                        help="Flip left/right hand assignment (try this if hands are swapped)")
    parser.add_argument("--no-voice", action="store_true")
    args = parser.parse_args()

    print("🥁 GESTURE.DJ starting up...\n")

    print("🔊 Loading audio samples...")
    sound = SoundEngine(args.samples_dir)
    sound.preload_all()
    print(f"   ✓ {sound.sample_count()} samples loaded\n")

    voice = None
    if args.elevenlabs_key and not args.no_voice:
        print("🎙  Connecting to ElevenLabs...")
        voice = ElevenLabsClient(args.elevenlabs_key)
        voice.prewarm([
            "Drum mode", "Violin mode",
            "Switching to drums", "Switching to violin",
        ])
        print("   ✓ Voice ready\n")
    else:
        print("   ℹ️  Voice disabled — pass --elevenlabs-key or set ELEVENLABS_API_KEY\n")

    overlay = Overlay()

    print("👋 Starting gesture detection...")
    engine = GestureEngine(
        camera_index=args.camera,
        sound=sound,
        voice=voice,
        overlay=overlay,
        mirror=args.mirror,
    )
    print("   ✓ Camera open\n")

    print("━" * 52)
    print("  Hands not swapping correctly? Run with:  --mirror")
    print("  Wrong camera?  Run with:  --camera 1  (or 2)")
    print("  Q to quit")
    print("━" * 52)
    print()

    if voice:
        threading.Thread(target=voice.speak, args=("Drum mode",), daemon=True).start()

    engine.run()

    print("\n👋 Goodbye!")
    sound.shutdown()


if __name__ == "__main__":
    main()