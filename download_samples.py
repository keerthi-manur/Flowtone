"""
download_samples.py
Generates drum + violin samples locally using numpy synthesis.
Run once before starting the app.
"""

import os
import wave
import numpy as np

SAMPLES_DIR = "samples"
SR = 44100  # sample rate

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


def save_wav(path, signal, sr=SR):
    signal = np.clip(signal, -1, 1)
    data   = (signal * 32767).astype(np.int16)
    stereo = np.column_stack([data, data])
    with wave.open(path, 'w') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(stereo.tobytes())


def t_arr(duration):
    return np.linspace(0, duration, int(SR * duration), endpoint=False)


# ── DRUM GENERATORS ──────────────────────────────────────────────

def make_kick():
    """808-style kick: pitch sweep from 150Hz → 40Hz with punchy transient"""
    t   = t_arr(0.55)
    # Frequency sweep
    freq = 150 * np.exp(-18 * t) + 40
    phase = np.cumsum(2 * np.pi * freq / SR)
    tone  = np.sin(phase)
    # Click transient at start
    click = np.exp(-80 * t) * 0.6
    # Body envelope
    env   = np.exp(-6 * t) * 0.95
    # Slight distortion for punch
    sig   = np.tanh((tone * env + click) * 1.5) / 1.5
    return sig * 0.9


def make_snare():
    """Snare: sharp crack tone + tuned noise burst, short decay"""
    t     = t_arr(0.35)
    # Tuned body (two tones for that snare character)
    body  = (np.sin(2*np.pi*185*t) * 0.5 +
             np.sin(2*np.pi*230*t) * 0.3)
    body_env = np.exp(-22 * t)
    # Noise component (the "snare wires")
    noise    = np.random.randn(len(t)) * 0.6
    noise_env = np.exp(-14 * t)
    # Combine with transient
    sig = body * body_env + noise * noise_env
    # Normalize and add click
    sig += np.exp(-120 * t) * 0.4
    return sig * 0.85


def make_hihat(open_hat=False):
    """Hi-hat: filtered white noise, very short (closed) or longer (open)"""
    dur   = 0.5 if open_hat else 0.09
    t     = t_arr(dur)
    noise = np.random.randn(len(t))
    # Simple high-pass: subtract smoothed version
    from numpy.fft import rfft, irfft, rfftfreq
    F     = rfft(noise)
    freqs = rfftfreq(len(noise), 1/SR)
    F[freqs < 6000] *= (freqs[freqs < 6000] / 6000) ** 2  # roll off below 6kHz
    noise = irfft(F, n=len(t))
    decay = 3 if open_hat else 60
    env   = np.exp(-decay * t)
    return noise * env * (0.55 if open_hat else 0.45)


def make_tom():
    """Floor tom: low pitch sweep, more resonance than kick"""
    t    = t_arr(0.45)
    freq = 100 * np.exp(-8 * t) + 55
    phase = np.cumsum(2 * np.pi * freq / SR)
    tone  = np.sin(phase) + 0.3 * np.sin(2 * phase)
    env   = np.exp(-7 * t)
    noise = np.random.randn(len(t)) * 0.05 * np.exp(-20 * t)
    return (tone * env + noise) * 0.88


def make_crash():
    """Crash cymbal: complex metallic noise, long ring"""
    t     = t_arr(1.8)
    noise = np.random.randn(len(t))
    # Multiple resonant frequencies for metallic character
    freqs = [214, 428, 635, 891, 1200, 1680, 2400]
    ring  = sum(np.sin(2*np.pi*f*t) * np.exp(-(2+i*0.3)*t)
                for i, f in enumerate(freqs))
    from numpy.fft import rfft, irfft, rfftfreq
    F     = rfft(noise)
    freqs_arr = rfftfreq(len(noise), 1/SR)
    F[freqs_arr < 3000] = 0
    noise = irfft(F, n=len(t))
    env   = np.exp(-1.8 * t)
    sig   = (noise * 0.6 + ring * 0.4) * env
    return sig * 0.7


def make_rimshot():
    """Rimshot: sharp high-pitched crack, very short"""
    t    = t_arr(0.18)
    tone = np.sin(2*np.pi*800*t) * 0.5 + np.sin(2*np.pi*1200*t) * 0.3
    env  = np.exp(-35 * t)
    noise = np.random.randn(len(t)) * 0.3 * np.exp(-40 * t)
    return (tone * env + noise) * 0.8


# ── VIOLIN GENERATOR ─────────────────────────────────────────────

def make_violin(freq, duration=2.5):
    """
    Violin-like additive synthesis with bow attack envelope.
    Strong odd harmonics, slight vibrato, bow noise.
    """
    t = t_arr(duration)

    # Additive harmonics (violin favors odd)
    harmonics = [(1,1.0),(2,0.4),(3,0.7),(4,0.15),(5,0.5),(6,0.08),(7,0.3)]
    sig = sum(amp * np.sin(2*np.pi*freq*h*t) for h, amp in harmonics)

    # Vibrato: 5Hz, subtle
    vibrato  = 1 + 0.007 * np.sin(2*np.pi*5.2*t)
    sig_vib  = sum(amp * np.sin(2*np.pi*freq*h*vibrato*t)
                   for h, amp in harmonics)

    # Blend non-vibrato (attack) into vibrato (sustain)
    blend_frames = int(0.25 * SR)
    blend = np.zeros(len(t))
    blend[:blend_frames] = np.linspace(0, 1, blend_frames)
    blend[blend_frames:] = 1.0
    sig = sig * (1-blend) + sig_vib * blend

    # Bow envelope: slow attack (string catches bow), sustain, release
    attack  = int(0.06 * SR)
    release = int(0.35 * SR)
    env     = np.ones(len(t)) * 0.88
    env[:attack]   = np.linspace(0, 0.88, attack)
    env[-release:] = np.linspace(0.88, 0, release)

    # Bow rosin noise (very subtle)
    bow_noise = np.random.randn(len(t)) * 0.025
    bow_env   = np.exp(-3 * t) * 0.5 + 0.5
    sig = (sig + bow_noise * bow_env) * env

    return sig * 0.82


# ── MAIN ─────────────────────────────────────────────────────────

def main():
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    print(f"\n📁 Generating samples into ./{SAMPLES_DIR}/\n")

    drums = {
        "kick":       make_kick,
        "snare":      make_snare,
        "hihat":      make_hihat,
        "hihat_open": lambda: make_hihat(open_hat=True),
        "tom":        make_tom,
        "crash":      make_crash,
        "rimshot":    make_rimshot,
    }

    print("🥁 Drums:")
    for name, fn in drums.items():
        path = os.path.join(SAMPLES_DIR, f"{name}.wav")
        if os.path.exists(path):
            print(f"  ↷ {name}.wav exists, skipping")
            continue
        sig = fn()
        save_wav(path, sig)
        print(f"  ✓ {name}")

    print("\n🎻 Violin:")
    for name, freq in VIOLIN_NOTES.items():
        path = os.path.join(SAMPLES_DIR, f"{name}.wav")
        if os.path.exists(path):
            print(f"  ↷ {name}.wav exists, skipping")
            continue
        sig = make_violin(freq)
        save_wav(path, sig)
        print(f"  ✓ {name}  ({freq:.1f} Hz)")

    total = len(os.listdir(SAMPLES_DIR))
    print(f"\n✅ Done — {total} files in ./{SAMPLES_DIR}/")
    print("   Tip: replace any .wav with a real recording using the same filename.\n")


if __name__ == "__main__":
    main()