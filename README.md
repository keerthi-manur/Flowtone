# 🎵 Flowtone

Play instruments with your hands. No controllers, no keyboard — just your webcam and gestures.

Flowtone uses real-time hand tracking to let you perform drums and violin using nothing but hand positions and finger gestures. Built with MediaPipe, OpenCV, and pygame.

---

## Modes

### 🥁 Drum Mode

Each hand plays a different kit:

| Gesture | Left hand | Right hand |
|---------|-----------|------------|
| ✊ Fist | rest position | rest position |
| ☝ 1 finger | Snare | Snare 2 |
| ✌ 2 fingers | Hi-Hat | Tambourine |
| 3 fingers | Bass Drum | Djembe |
| 🖐 Open palm | Hand Cymbals | Clash Cymbals |

Right hand height controls volume.

### 🎻 Violin Mode

| Action | Effect |
|--------|--------|
| Left hand height | Pitch (A4 → G5) |
| Right hand height | Volume / bow pressure |
| 👋 Wave left hand | Vibrato |
| ✊ Fist | Mute |

Hold a note still and it transitions to a sustained bow sound. Slide through notes quickly for a staccato effect.

### Switching modes

Hold both 👍 thumbs up simultaneously to switch between drum and violin mode.

---

## Setup

### 1. Install dependencies

```bash
pip install mediapipe opencv-python pygame numpy requests
```

> macOS: grant camera access to Terminal in System Settings → Privacy & Security → Camera.

### 2. Add your samples

Place your audio files in a `samples/` folder. Supported formats: `.wav`, `.mp3`

**Drums** — name your files:
`snare.mp3`, `hihat.mp3`, `bass_drum.mp3`, `hand_cymbals.mp3`, `snare2.mp3`, `tambourine.mp3`, `djembe.mp3`, `clash_cymbals.mp3`

**Violin** — name your files:
`violin_A4.mp3`, `violin_B4.mp3`, ... `violin_G5.mp3`

For sustained bow sounds, add long versions:
`violin_A4_long.mp3`, `violin_B4_long.mp3`, ... `violin_G5_long.mp3`

If you don't have samples, run the synthesizer to generate basic ones:
```bash
python download_samples.py
```

### 3. (Optional) ElevenLabs voice announcements

```bash
export ELEVENLABS_API_KEY=your_key_here
```

---

## Run

```bash
python main.py
```

A window opens with your webcam feed. Hold your hands up in frame.

---

## Tips

- Keep hands between shoulder and waist height — the full vertical range maps to pitch and volume
- Use a fist as your neutral/rest position between drum hits
- Better lighting = more accurate tracking (face a window)
- Plain background behind your hands helps detection
- Press `Q` to quit

---

## File structure

```
flowtone/
├── main.py                # entry point
├── gesture_engine.py      # MediaPipe tracking + gesture classification
├── sound_engine.py        # pygame audio management
├── elevenlabs_client.py   # TTS voice announcements
├── overlay.py             # OpenCV HUD
├── download_samples.py    # generates synthesized samples as fallback
└── samples/               # your audio files go here
```

---

## Troubleshooting

**Wrong camera** — try `--camera 1` or `--camera 2`

**Hands swapped** — run with `--no-mirror` to toggle hand assignment

**No sound** — check that `./samples/` exists and has audio files in it

**Gestures not detecting well** — better lighting, keep hands 30–60cm from camera, avoid busy backgrounds