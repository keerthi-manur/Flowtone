# 🥁 GESTURE.DJ

Play drums and violin with your hands. No keyboard. No mouse.  
MediaPipe hand tracking → gesture classification → real-time audio.

---

## Setup (do this once)

### 1. Install dependencies

```bash
pip install mediapipe opencv-python pygame numpy requests
```

> On macOS you may need to grant camera access to Terminal/iTerm in  
> System Settings → Privacy & Security → Camera.

### 2. Generate audio samples

```bash
python download_samples.py
```

This synthesizes drum hits and violin notes locally — no internet needed.  
You can replace any `.wav` in `./samples/` with your own real recordings.

### 3. (Optional) Add your ElevenLabs key

```bash
export ELEVENLABS_API_KEY=your_key_here
```

Or pass it as a flag: `--elevenlabs-key your_key_here`

---

## Run

```bash
python main.py
```

A window opens with your webcam feed. Hold your hands up in frame.

---

## Controls

| Hand | What it controls |
|------|-----------------|
| Left hand height | Pitch (violin) |
| Right hand height | Volume (both modes) |
| Left hand gesture | Drum hit / bowing style |
| Thumbs up (either hand) | Switch mode |

### Drum Mode gestures (left hand)

| Gesture | Sound |
|---------|-------|
| ✊ Fist | Kick drum |
| ☝ 1 finger | Snare |
| ✌ 2 fingers | Hi-hat (closed) |
| 3 fingers | Tom |
| 🖐 Open palm | Crash cymbal |
| 👌 Pinch | Rimshot |
| 👍 Thumbs up | → Switch to Violin |

Right hand also triggers:  
✊ Fist = Kick, ☝ = Hi-hat open, 🖐 = Crash

### Violin Mode gestures

| Action | Effect |
|--------|--------|
| Left hand height | Note (A3 → D5, pentatonic) |
| Right hand height | Volume / bow pressure |
| 👋 Wave left hand | Vibrato |
| ✊ Fist (left) | Mute |
| 👍 Thumbs up | → Switch to Drums |

---

## Tips

- **Keep hands between shoulder and waist height** — the full Y range maps to pitch/volume
- **Drum mode works best** when you make deliberate, clean gestures and hold them briefly
- **Violin mode** is continuous — your note changes as you move your hand up/down
- If gestures feel jittery, try better lighting (face a window)
- Press `Q` to quit

---

## File structure

```
gesture_dj/
├── main.py              # entry point
├── gesture_engine.py    # MediaPipe tracking + gesture classification
├── sound_engine.py      # pygame audio management
├── elevenlabs_client.py # TTS voice announcements
├── overlay.py           # OpenCV HUD
├── download_samples.py  # generates audio samples
└── samples/             # .wav files (created by download_samples.py)
    ├── kick.wav
    ├── snare.wav
    ├── ...
    ├── violin_A3.wav
    └── violin_D5.wav
```

---

## Troubleshooting

**Camera not found**  
Try `--camera 1` or `--camera 2` if you have multiple cameras.

**No sound**  
Make sure you ran `python download_samples.py` first.  
Check that `./samples/` has `.wav` files in it.

**Gestures not detecting well**  
- Better lighting helps a lot
- Keep hands 30–60 cm from camera
- Plain background behind hands (avoid busy patterns)

**macOS camera permission denied**  
System Settings → Privacy & Security → Camera → enable Terminal (or your app)
