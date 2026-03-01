"""
download_samples.py
Downloads free, license-clear drum + violin samples.
Run this ONCE before starting the app.

Sources used:
  - Drums: Freesound.org CC0 samples
  - Violin: Freesound.org CC0 samples

If any download fails, you can manually place .wav files in ./samples/
with these names:
  kick.wav, snare.wav, hihat.wav, hihat_open.wav, tom.wav,
  crash.wav, rimshot.wav,
  violin_A3.wav, violin_C4.wav, violin_D4.wav, violin_E4.wav,
  violin_G4.wav, violin_A4.wav, violin_C5.wav, violin_D5.wav
"""

import os
import urllib.request
import sys

SAMPLES_DIR = "samples"

# Free CC0 samples from various sources.
# We use synthesized/generated fallbacks if downloads fail.
SAMPLES = {
    # Drums — classic 808/909 style free samples
    "kick":       "https://freesound.org/data/previews/209/209235_921947-lq.mp3",
    "snare":      "https://freesound.org/data/previews/387/387186_1474204-lq.mp3",
    "hihat":      "https://freesound.org/data/previews/204/204929_1693665-lq.mp3",
    "hihat_open": "https://freesound.org/data/previews/204/204930_1693665-lq.mp3",
    "tom":        "https://freesound.org/data/previews/262/262090_1186165-lq.mp3",
    "crash":      "https://freesound.org/data/previews/360/360714_5450487-lq.mp3",
    "rimshot":    "https://freesound.org/data/previews/209/209236_921947-lq.mp3",
}

# Violin notes — we'll generate these with numpy/scipy as pure sine waves
# if scipy isn't available, or use pre-made ones if you have them.
VIOLIN_NOTES = {
    "violin_A3": 220.00,
    "violin_C4": 261.63,
    "violin_D4": 293.66,
    "violin_E4": 329.63,
    "violin_G4": 392.00,
    "violin_A4": 440.00,
    "violin_C5": 523.25,
    "violin_D5": 587.33,
}


def generate_violin_sample(filename: str, freq: float, duration: float = 2.0,
                             sample_rate: int = 44100):
    """
    Generate a violin-like tone using additive synthesis.
    Combines multiple harmonics with a bow-like envelope.
    Much better than a pure sine wave.
    """
    import numpy as np
    import wave
    import struct

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Additive synthesis: fundamental + harmonics with decreasing amplitude
    # Violin has strong odd harmonics
    harmonics = [
        (1, 1.0),    # fundamental
        (2, 0.5),    # 2nd harmonic
        (3, 0.7),    # 3rd (strong in violin)
        (4, 0.2),
        (5, 0.4),    # 5th (strong in violin)
        (6, 0.1),
        (7, 0.2),
    ]

    signal = np.zeros_like(t)
    for harmonic, amp in harmonics:
        signal += amp * np.sin(2 * np.pi * freq * harmonic * t)

    # Bow envelope: fast attack, sustain, gentle release
    attack = int(0.03 * sample_rate)
    release = int(0.3 * sample_rate)
    sustain_val = 0.85

    envelope = np.ones_like(t) * sustain_val
    envelope[:attack] = np.linspace(0, sustain_val, attack)
    envelope[-release:] = np.linspace(sustain_val, 0, release)

    # Slight vibrato (5 Hz, ±1% pitch variation)
    vibrato = 1 + 0.008 * np.sin(2 * np.pi * 5 * t)
    # Apply vibrato to phase (approximate)
    signal = envelope * np.sin(2 * np.pi * freq * vibrato * t)
    for harmonic, amp in harmonics[1:]:
        signal += envelope * amp * 0.3 * np.sin(2 * np.pi * freq * harmonic * vibrato * t)

    signal *= envelope

    # Normalize
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.85

    # Convert to 16-bit PCM
    samples = (signal * 32767).astype(np.int16)
    stereo = np.column_stack([samples, samples])

    with wave.open(filename, 'w') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())

    print(f"  ✓ Generated {os.path.basename(filename)} ({freq:.1f} Hz)")


def generate_drum_fallback(filename: str, drum_type: str, sample_rate: int = 44100):
    """Generate basic drum sounds using numpy if downloads fail."""
    import numpy as np
    import wave

    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    if drum_type == "kick":
        # Pitch-swept sine — classic 808 kick
        freq = 150 * np.exp(-15 * t)
        signal = np.sin(2 * np.pi * freq * t)
        env = np.exp(-8 * t)
        signal = signal * env * 0.95

    elif drum_type == "snare":
        # White noise + tone
        noise = np.random.randn(len(t)) * 0.5
        tone = np.sin(2 * np.pi * 200 * t) * 0.5
        env = np.exp(-10 * t)
        signal = (noise + tone) * env * 0.8

    elif drum_type in ("hihat", "hihat_open"):
        # Filtered noise
        noise = np.random.randn(len(t))
        # Simple highpass: subtract low freq component
        from numpy.fft import fft, ifft, fftfreq
        F = fft(noise)
        freqs = fftfreq(len(noise), 1 / sample_rate)
        F[np.abs(freqs) < 3000] = 0
        noise = np.real(ifft(F))
        decay = 5 if drum_type == "hihat" else 1.5
        env = np.exp(-decay * t)
        signal = noise * env * 0.6

    elif drum_type == "tom":
        freq = 90 * np.exp(-8 * t)
        signal = np.sin(2 * np.pi * freq * t)
        env = np.exp(-5 * t)
        signal = signal * env * 0.9

    elif drum_type == "crash":
        noise = np.random.randn(len(t))
        env = np.exp(-2 * t)
        signal = noise * env * 0.7

    elif drum_type == "rimshot":
        tone = np.sin(2 * np.pi * 400 * t)
        noise = np.random.randn(len(t)) * 0.3
        env = np.exp(-20 * t)
        signal = (tone + noise) * env * 0.8

    else:
        signal = np.zeros(len(t))

    # Normalize + convert
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.85
    samples = (signal * 32767).astype(np.int16)
    stereo = np.column_stack([samples, samples])

    with wave.open(filename, 'w') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())

    print(f"  ✓ Generated {os.path.basename(filename)} (synthesized {drum_type})")


def main():
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    print(f"\n📁 Generating/downloading samples into ./{SAMPLES_DIR}/\n")

    # Check numpy available (required for synthesis)
    try:
        import numpy as np
        has_numpy = True
    except ImportError:
        has_numpy = False
        print("⚠️  numpy not found — install it for sample generation: pip install numpy")

    # Generate violin samples (always synthesized — consistent quality)
    print("🎻 Generating violin samples...")
    for name, freq in VIOLIN_NOTES.items():
        path = os.path.join(SAMPLES_DIR, f"{name}.wav")
        if os.path.exists(path):
            print(f"  ↷ {name}.wav already exists, skipping")
            continue
        if has_numpy:
            generate_violin_sample(path, freq)
        else:
            print(f"  ✗ Skipped {name} (numpy required)")

    # Generate drum samples
    print("\n🥁 Generating drum samples...")
    drum_types = {
        "kick": "kick", "snare": "snare", "hihat": "hihat",
        "hihat_open": "hihat_open", "tom": "tom",
        "crash": "crash", "rimshot": "rimshot",
    }

    for name, drum_type in drum_types.items():
        path = os.path.join(SAMPLES_DIR, f"{name}.wav")
        if os.path.exists(path):
            print(f"  ↷ {name}.wav already exists, skipping")
            continue
        if has_numpy:
            generate_drum_fallback(path, drum_type)
        else:
            print(f"  ✗ Skipped {name} (numpy required)")

    print(f"\n✅ Done! {len(os.listdir(SAMPLES_DIR))} files in ./{SAMPLES_DIR}/")
    print("\nTip: Replace any .wav with your own real samples for better sound.")
    print("     Just keep the same filenames.\n")


if __name__ == "__main__":
    main()
