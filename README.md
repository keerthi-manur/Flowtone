# 🎵 Flowtone

Play instruments with your hands. No controllers, no keyboard — just your webcam and gestures.

Flowtone uses real-time hand tracking to let you perform drums, violin, and flute using nothing but hand positions and finger gestures. Built with MediaPipe, OpenCV, and pygame. Audio samples from the Philharmonia Orchestra.

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

### 🎵 Flute Mode

Hold your hands up with palms facing you, fingers pointing up. Curl fingers progressively to play notes:

| Left hand (lower octave) | Right hand (upper octave) |
|--------------------------|---------------------------|
| Index down → A5 | Index down → A6 |
| Index + middle → B5 | Index + middle → B6 |
| Index + middle + ring → C5 | Index + middle + ring → C6 |
| All four fingers → D5 | All four fingers → D6 |

Open hand = rest/silence.

### Switching modes

Hold both 👍 thumbs up simultaneously, or press `M`, to cycle through modes: Drums → Violin → Flute → Drums.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/flowtone.git
cd flowtone
```

### 2. Install dependencies

```bash
pip install mediapipe opencv-python pygame numpy requests
```

> macOS: grant camera access to Terminal in System Settings → Privacy & Security → Camera.

### 3. Add your samples

Place your audio files in a `samples/` folder. Supported formats: `.wav`, `.mp3`

#### Quick start — generate placeholder samples

If you just want to get up and running, run the synthesizer to generate basic sounds:

```bash
python download_samples.py
```

This generates simple drum hits and violin notes locally — no internet needed. They're functional but not the most expressive.

#### For the full experience — Philharmonia Orchestra samples

This project was built using samples from the **Philharmonia Orchestra Sound Samples** library, which are free to download for anyone:

> [philharmonia.co.uk/explore/sound_samples](https://www.philharmonia.co.uk/explore/sound_samples)

Search by instrument, download the notes you need, and rename them to match the filenames below.

**Drums** — name your files:
`snare.mp3`, `hihat.mp3`, `bass_drum.mp3`, `hand_cymbals.mp3`, `snare2.mp3`, `tambourine.mp3`, `djembe.mp3`, `clash_cymbals.mp3`

For drums, free one-shot samples are also available on [freesound.org](https://freesound.org) (filter by CC0 license) and [99sounds.org](https://99sounds.org).

**Violin** — name your files:
`violin_A4.mp3`, `violin_B4.mp3`, `violin_C4.mp3`, `violin_D4.mp3`, `violin_E4.mp3`, `violin_F4.mp3`, `violin_G4.mp3` and the same for the 5th octave (`violin_A5.mp3` ... `violin_G5.mp3`)

For sustained bow sounds, add long versions with `_long` suffix:
`violin_A4_long.mp3`, `violin_B4_long.mp3`, ... `violin_G5_long.mp3`

**Flute** — name your files:
```
flute_A5.mp3
flute_B5.mp3
flute_C5.mp3
flute_D5.mp3
flute_A6_long.mp3
flute_B6_long.mp3
flute_C6_long.mp3
flute_D6_long.mp3
```

### 4. (Optional) ElevenLabs voice announcements

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

## Controls

| Key | Action |
|-----|--------|
| `M` | Switch mode |
| `Q` | Quit |
| Both thumbs up | Switch mode |

---

## Tips

- Keep hands between shoulder and waist height — the full vertical range maps to pitch and volume
- In drum mode, use a fist as your neutral/rest position between hits
- In flute mode, hold palms facing you with fingers pointing up
- Better lighting = more accurate tracking (face a window)
- Plain background behind your hands helps detection

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
└── samples/               # your audio files go here (not included)
```

---

## Troubleshooting

**Wrong camera** — try `--camera 1` or `--camera 2`

**Hands swapped** — run with `--no-mirror` to toggle hand assignment

**No sound** — check that `./samples/` exists and has audio files in it

**Gestures not detecting well** — better lighting, keep hands 30–60cm from camera, avoid busy backgrounds

**macOS camera permission denied** — System Settings → Privacy & Security → Camera → enable Terminal