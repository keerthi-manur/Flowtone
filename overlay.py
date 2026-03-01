"""
overlay.py
Clean HUD drawn on top of the OpenCV camera frame.
Always shows mode, current gesture, and a readable legend panel.
"""

import cv2
import numpy as np

# BGR colors
GREEN  = (0, 255, 160)
BLUE   = (255, 180, 0)
ORANGE = (53, 130, 255)
WHITE  = (230, 230, 230)
GRAY   = (140, 150, 155)
DIM    = (70, 85, 90)
BLACK  = (0, 0, 0)
YELLOW = (0, 220, 255)

PENTATONIC = ["A3","C4","D4","E4","G4","A4","C5","D5"]

DRUM_LEGEND = [
    ("LEFT HAND",  ""),
    ("✊ Fist",     "Kick"),
    ("☝ 1 finger", "Snare"),
    ("✌ 2 fingers","Hi-Hat"),
    ("3 fingers",  "Tom"),
    ("🖐 Palm",    "Crash"),
    ("👌 Pinch",   "Rimshot"),
    ("",           ""),
    ("RIGHT HAND", ""),
    ("✊ Fist",    "Kick"),
    ("☝ 1 finger", "Hi-Hat Open"),
    ("🖐 Palm",    "Crash"),
    ("",           ""),
    ("👍 Thumbs",  "→ Violin mode"),
]

VIOLIN_LEGEND = [
    ("LEFT HAND",  ""),
    ("↕ Height",   "Pitch"),
    ("👋 Wave",    "Vibrato"),
    ("✊ Fist",    "Mute"),
    ("",           ""),
    ("RIGHT HAND", ""),
    ("↕ Height",   "Volume"),
    ("",           ""),
    ("👍 Thumbs",  "→ Drum mode"),
]


def _alpha_rect(frame, x, y, w, h, color=(0,0,0), alpha=0.55):
    sub = frame[y:y+h, x:x+w]
    rect = np.full_like(sub, color, dtype=np.uint8)
    cv2.addWeighted(rect, alpha, sub, 1-alpha, 0, sub)
    frame[y:y+h, x:x+w] = sub


def _text(frame, txt, x, y, color=WHITE, scale=0.45, thick=1):
    # Black outline for readability on any background
    cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, BLACK, thick+2, cv2.LINE_AA)
    cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def _vbar(frame, x, y, w, h, value, color):
    """Vertical meter bar, value 0-1"""
    cv2.rectangle(frame, (x, y), (x+w, y+h), (15,20,25), -1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), DIM, 1)
    fill = int(h * max(0, min(1, value)))
    if fill > 0:
        cv2.rectangle(frame, (x, y+h-fill), (x+w, y+h), color, -1)


class Overlay:
    LEGEND_W = 185

    def draw(self, frame, engine):
        fh, fw = frame.shape[:2]
        mode   = engine.mode.value.upper()
        lh     = engine.left_hand
        rh     = engine.right_hand
        is_drum = mode == "DRUMS"

        # ── RIGHT SIDE LEGEND PANEL ──────────────────────────────
        lx = fw - self.LEGEND_W
        legend = DRUM_LEGEND if is_drum else VIOLIN_LEGEND
        panel_h = len(legend) * 17 + 30
        _alpha_rect(frame, lx, 0, self.LEGEND_W, panel_h, (5,10,15), alpha=0.72)
        cv2.rectangle(frame, (lx, 0), (fw-1, panel_h),
                      GREEN if is_drum else BLUE, 1)

        # Mode header
        mode_color = GREEN if is_drum else BLUE
        _text(frame, f"MODE: {mode}", lx+8, 18, mode_color, scale=0.5, thick=1)
        cv2.line(frame, (lx+6, 22), (fw-6, 22), mode_color, 1)

        cy = 38
        for left_col, right_col in legend:
            if left_col == "" and right_col == "":
                cy += 6
                continue
            # Section headers (right_col empty = header)
            if right_col == "":
                _text(frame, left_col, lx+8, cy, GRAY, scale=0.36, thick=1)
            else:
                _text(frame, left_col, lx+8,  cy, WHITE, scale=0.38)
                _text(frame, right_col, lx+90, cy, YELLOW, scale=0.38)
            cy += 17

        # ── TOP BAR ──────────────────────────────────────────────
        _alpha_rect(frame, 0, 0, lx, 38, (0,0,0), alpha=0.5)

        # FPS
        _text(frame, f"{engine.fps} FPS", 8, 22, DIM, scale=0.38)

        # Hint
        hint = "Q to quit  |  Thumbs up = switch mode"
        _text(frame, hint, 70, 22, DIM, scale=0.36)

        # ── LEFT HAND BOX ────────────────────────────────────────
        if lh.landmarks:
            _alpha_rect(frame, 4, 44, 160, 76, (0,0,0), alpha=0.5)
            cv2.rectangle(frame, (4,44), (164,120), GREEN, 1)
            _text(frame, "LEFT", 10, 59, GREEN, scale=0.4)

            action = engine.left_label
            _text(frame, action, 10, 82, WHITE, scale=0.62, thick=1)

            # Pitch note in violin mode
            if not is_drum:
                idx  = int((1.0 - lh.y_norm) * (len(PENTATONIC)-1))
                idx  = max(0, min(len(PENTATONIC)-1, idx))
                note = PENTATONIC[idx]
                _text(frame, note, 10, 108, GREEN, scale=0.7, thick=2)
                if lh.is_waving:
                    _text(frame, "~ vibrato", 70, 108, ORANGE, scale=0.35)

            # Pitch bar
            _vbar(frame, 140, 48, 16, 66, 1.0 - lh.y_norm, GREEN)
            _text(frame, "↕", 141, 122, DIM, scale=0.32)

        else:
            _alpha_rect(frame, 4, 44, 160, 30, (0,0,0), alpha=0.4)
            _text(frame, "LEFT — not detected", 10, 63, DIM, scale=0.36)

        # ── RIGHT HAND BOX ───────────────────────────────────────
        rx = lx - 168
        if rh.landmarks:
            _alpha_rect(frame, rx, 44, 160, 76, (0,0,0), alpha=0.5)
            cv2.rectangle(frame, (rx,44), (rx+160,120), BLUE, 1)
            _text(frame, "RIGHT", rx+8, 59, BLUE, scale=0.4)

            action = engine.right_label
            _text(frame, action, rx+8, 82, WHITE, scale=0.62, thick=1)

            vol = int((1.0 - rh.y_norm) * 100)
            _text(frame, f"vol {vol}%", rx+8, 108, BLUE, scale=0.42)

            # Volume bar
            _vbar(frame, rx+138, 48, 16, 66, 1.0 - rh.y_norm, BLUE)
            _text(frame, "↕", rx+139, 122, DIM, scale=0.32)

        else:
            _alpha_rect(frame, rx, 44, 160, 30, (0,0,0), alpha=0.4)
            _text(frame, "RIGHT — not detected", rx+8, 63, DIM, scale=0.36)

        # ── CENTER DIVIDER ───────────────────────────────────────
        cv2.line(frame, (fw//2, 36), (fw//2, fh),
                 (25, 35, 40), 1, cv2.LINE_AA)