"""
overlay.py
Draws the HUD on top of the OpenCV camera frame.
Shows mode, gesture labels, pitch/volume meters, active note.
"""

import cv2
import numpy as np
import time


# Colors (BGR for OpenCV)
GREEN  = (0, 255, 160)
BLUE   = (255, 180, 0)
ORANGE = (53, 107, 255)
WHITE  = (220, 220, 220)
DIM    = (80, 100, 110)
BLACK  = (0, 0, 0)
RED    = (60, 60, 220)


def _put_text(frame, text, pos, color=WHITE, scale=0.45, thickness=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _rect(frame, x, y, w, h, color, alpha=0.35):
    """Semi-transparent filled rectangle"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _bar(frame, x, y, w, h, value, color, label=""):
    """Draw a vertical meter bar (value 0.0–1.0)"""
    # Background
    cv2.rectangle(frame, (x, y), (x + w, y + h), (20, 30, 35), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), DIM, 1)
    # Fill
    fill_h = int(h * value)
    if fill_h > 0:
        cv2.rectangle(frame, (x, y + h - fill_h), (x + w, y + h), color, -1)
    if label:
        _put_text(frame, label, (x, y - 4), DIM, scale=0.35)


# Pentatonic note names for violin mode
PENTATONIC = ["A3", "C4", "D4", "E4", "G4", "A4", "C5", "D5"]

# Drum gesture map
DRUM_MAP = {
    "fist":        "KICK",
    "one_finger":  "SNARE",
    "two_fingers": "HI-HAT",
    "three_fingers": "TOM",
    "open_palm":   "CRASH",
    "pinch":       "RIMSHOT",
    "thumbs_up":   "→ VIOLIN",
    "neutral":     "—",
    "wave":        "—",
}

VIOLIN_MAP = {
    "fist":       "MUTE",
    "open_palm":  "BOW",
    "wave":       "VIBRATO",
    "thumbs_up":  "→ DRUMS",
    "neutral":    "BOW",
    "one_finger": "BOW",
    "two_fingers": "BOW",
    "three_fingers": "BOW",
    "pinch":      "MUTE",
}


class Overlay:
    def __init__(self):
        self._note_flash_time = 0
        self._last_note = ""

    def draw(self, frame, engine):
        h, w = frame.shape[:2]
        mode = engine.mode.value.upper()
        lh = engine.left_hand
        rh = engine.right_hand

        # ── TOP BAR ─────────────────────────────────────────
        _rect(frame, 0, 0, w, 36, BLACK, alpha=0.55)

        # Mode pill
        mode_color = GREEN if mode == "DRUMS" else BLUE
        cv2.rectangle(frame, (10, 6), (110, 30), mode_color, -1, cv2.LINE_AA)
        cv2.putText(frame, f"  {mode}  ", (14, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 2, cv2.LINE_AA)

        # FPS
        _put_text(frame, f"{engine.fps} FPS", (w - 70, 22), DIM, scale=0.38)

        # Tip
        tip = "THUMBS UP to switch mode" if lh.gesture.value != "thumbs_up" else ""
        _put_text(frame, tip, (120, 22), DIM, scale=0.35)

        # ── LEFT HAND PANEL ──────────────────────────────────
        if lh.landmarks:
            _rect(frame, 4, 44, 150, 90, BLACK, alpha=0.45)
            cv2.rectangle(frame, (4, 44), (154, 134), GREEN, 1)
            _put_text(frame, "LEFT HAND", (10, 58), GREEN, scale=0.38)

            g_name = lh.gesture.value
            if lh.is_waving:
                g_name = "wave"
            g_label = (DRUM_MAP if mode == "DRUMS" else VIOLIN_MAP).get(g_name, "—")
            _put_text(frame, g_label, (10, 80), WHITE, scale=0.55, thickness=1)

            # Y meter (pitch in violin, just height indicator in drums)
            pitch_val = 1.0 - lh.y_norm
            _bar(frame, 130, 48, 18, 80, pitch_val, GREEN,
                 "PITCH" if mode == "VIOLIN" else "↕")

            # In violin mode, show current note
            if mode == "VIOLIN":
                note_idx = int(pitch_val * (len(PENTATONIC) - 1))
                note_idx = max(0, min(len(PENTATONIC) - 1, note_idx))
                note = PENTATONIC[note_idx]
                _put_text(frame, note, (10, 108), GREEN, scale=0.7, thickness=2)

            # Vibrato indicator
            if lh.is_waving and mode == "VIOLIN":
                _put_text(frame, "〜 VIBRATO", (10, 125), ORANGE, scale=0.32)

        else:
            _rect(frame, 4, 44, 150, 36, BLACK, alpha=0.3)
            _put_text(frame, "LEFT HAND — none", (10, 65), DIM, scale=0.38)

        # ── RIGHT HAND PANEL ─────────────────────────────────
        if rh.landmarks:
            rx = w - 160
            _rect(frame, rx - 4, 44, 160, 90, BLACK, alpha=0.45)
            cv2.rectangle(frame, (rx - 4, 44), (w - 4, 134), BLUE, 1)
            _put_text(frame, "RIGHT HAND", (rx, 58), BLUE, scale=0.38)

            g_name = rh.gesture.value
            g_label = (DRUM_MAP if mode == "DRUMS" else VIOLIN_MAP).get(g_name, "—")
            _put_text(frame, g_label, (rx, 80), WHITE, scale=0.55)

            # Volume bar
            vol_val = 1.0 - rh.y_norm
            _bar(frame, w - 26, 48, 18, 80, vol_val, BLUE, "VOL")

            vol_pct = int(vol_val * 100)
            _put_text(frame, f"{vol_pct}%", (rx, 108), BLUE, scale=0.5)

        else:
            rx = w - 160
            _rect(frame, rx - 4, 44, 160, 36, BLACK, alpha=0.3)
            _put_text(frame, "RIGHT HAND — none", (rx, 65), DIM, scale=0.38)

        # ── BOTTOM LEGEND ────────────────────────────────────
        _rect(frame, 0, h - 30, w, 30, BLACK, alpha=0.55)
        if mode == "DRUMS":
            legend = "✊KICK  ☝SNARE  ✌HI-HAT  THREE:TOM  🖐CRASH  PINCH:RIMSHOT"
        else:
            legend = "LEFT HEIGHT:PITCH  RIGHT HEIGHT:VOL  WAVE:VIBRATO  ✊MUTE"
        _put_text(frame, legend, (10, h - 10), DIM, scale=0.33)

        # ── CENTER DIVIDER (subtle) ───────────────────────────
        cv2.line(frame, (w // 2, 40), (w // 2, h - 35),
                 (30, 45, 50), 1, cv2.LINE_AA)
