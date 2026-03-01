"""
gesture_engine.py
MediaPipe hand tracking → gesture classification → sound/voice triggers
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Mode(Enum):
    DRUMS = "drums"
    VIOLIN = "violin"


class Gesture(Enum):
    FIST = "fist"
    ONE_FINGER = "one_finger"
    TWO_FINGERS = "two_fingers"
    THREE_FINGERS = "three_fingers"
    OPEN_PALM = "open_palm"
    THUMBS_UP = "thumbs_up"
    WAVE = "wave"
    PINCH = "pinch"
    NEUTRAL = "neutral"


@dataclass
class HandState:
    """Tracks state for one hand across frames"""
    landmarks: list = field(default_factory=list)
    gesture: Gesture = Gesture.NEUTRAL
    prev_gesture: Gesture = Gesture.NEUTRAL
    gesture_frames: int = 0          # how many consecutive frames this gesture held
    y_norm: float = 0.5              # normalized Y position (0=top, 1=bottom)
    x_norm: float = 0.5             # normalized X position
    prev_x: float = 0.5
    wave_history: list = field(default_factory=list)
    is_waving: bool = False
    last_trigger_time: float = 0.0  # debounce


class GestureEngine:
    # Tuning constants
    GESTURE_CONFIRM_FRAMES = 3    # frames before a gesture fires
    TRIGGER_DEBOUNCE = 0.18       # seconds between same-gesture triggers (drums)
    VIOLIN_UPDATE_INTERVAL = 0.05 # seconds between violin note updates
    THUMBS_DEBOUNCE = 1.0         # seconds between mode switches
    WAVE_WINDOW = 12              # frames for wave detection
    WAVE_THRESHOLD = 0.07         # x-delta threshold for wave

    def __init__(self, camera_index, sound, voice, overlay):
        self.sound = sound
        self.voice = voice
        self.overlay = overlay
        self.mode = Mode.DRUMS

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_style = mp.solutions.drawing_styles

        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.left_hand = HandState()
        self.right_hand = HandState()

        self.last_thumbs_time = 0.0
        self.last_violin_time = 0.0
        self.last_violin_note = -1

        self.frame_count = 0
        self.fps = 0
        self.fps_time = time.time()

    # ── LANDMARK HELPERS ──────────────────────────────────────────

    def _tip(self, lm, idx):
        """Return (x, y) normalized for a landmark index"""
        l = lm[idx]
        return l.x, l.y

    def _finger_extended(self, lm, tip_idx, pip_idx):
        """True if fingertip is above its PIP joint (finger up)"""
        return lm[tip_idx].y < lm[pip_idx].y

    def _count_extended_fingers(self, lm):
        """Count how many of the 4 fingers (not thumb) are extended"""
        fingers = [
            (8, 6),   # index tip, pip
            (12, 10), # middle
            (16, 14), # ring
            (20, 18), # pinky
        ]
        return sum(self._finger_extended(lm, t, p) for t, p in fingers)

    def _thumb_extended(self, lm, handedness):
        """Check if thumb tip is to the side of thumb IP joint"""
        tip_x = lm[4].x
        ip_x = lm[3].x
        # For right hand, tip should be to the left of IP to be extended
        if handedness == "Right":
            return tip_x < ip_x - 0.03
        else:
            return tip_x > ip_x + 0.03

    def _thumbs_up(self, lm, handedness):
        """Thumb up + all other fingers curled"""
        if not self._thumb_extended(lm, handedness):
            return False
        # Thumb tip must be clearly above wrist
        if lm[4].y > lm[0].y - 0.05:
            return False
        # Other fingers curled
        return self._count_extended_fingers(lm) == 0

    def _pinch_distance(self, lm):
        """Distance between thumb tip and index tip"""
        tx, ty = lm[4].x, lm[4].y
        ix, iy = lm[8].x, lm[8].y
        return ((tx - ix)**2 + (ty - iy)**2) ** 0.5

    def classify_gesture(self, lm, handedness) -> Gesture:
        """Classify a hand's landmarks into a Gesture enum"""
        n_fingers = self._count_extended_fingers(lm)
        thumb_up = self._thumbs_up(lm, handedness)

        if thumb_up:
            return Gesture.THUMBS_UP

        # Pinch check
        if self._pinch_distance(lm) < 0.05 and n_fingers <= 1:
            return Gesture.PINCH

        if n_fingers == 0:
            return Gesture.FIST
        elif n_fingers == 1:
            return Gesture.ONE_FINGER
        elif n_fingers == 2:
            return Gesture.TWO_FINGERS
        elif n_fingers == 3:
            return Gesture.THREE_FINGERS
        else:
            return Gesture.OPEN_PALM

    def _detect_wave(self, hand: HandState) -> bool:
        """Detect left-right waving motion from wrist x position history"""
        if len(hand.wave_history) < self.WAVE_WINDOW:
            return False
        history = hand.wave_history[-self.WAVE_WINDOW:]
        # Count direction changes
        changes = 0
        for i in range(1, len(history) - 1):
            prev_dir = history[i] - history[i-1]
            next_dir = history[i+1] - history[i]
            if abs(history[i] - history[i-1]) > self.WAVE_THRESHOLD:
                if (prev_dir > 0) != (next_dir > 0):
                    changes += 1
        return changes >= 2

    # ── HAND ASSIGNMENT ──────────────────────────────────────────

    def _assign_hands(self, results):
        """
        Map detected hands to left/right HandState objects.
        MediaPipe handedness is mirrored for selfie view.
        """
        self.left_hand.landmarks = []
        self.right_hand.landmarks = []

        if not results.multi_hand_landmarks:
            return

        for hand_lm, hand_info in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            lm = hand_lm.landmark
            label = hand_info.classification[0].label  # "Left" or "Right"
            # In selfie/mirror view, MediaPipe flips labels — we un-flip:
            # "Right" from MediaPipe = person's left hand (from camera's perspective)
            if label == "Right":
                self.left_hand.landmarks = lm
                self.left_hand.x_norm = lm[0].x
                self.left_hand.y_norm = lm[0].y
            else:
                self.right_hand.landmarks = lm
                self.right_hand.x_norm = lm[0].x
                self.right_hand.y_norm = lm[0].y

    # ── GESTURE → ACTION ─────────────────────────────────────────

    def _process_hand_gesture(self, hand: HandState, handedness: str, is_left: bool):
        """Update gesture state and detect new gestures"""
        if not hand.landmarks:
            hand.gesture = Gesture.NEUTRAL
            hand.gesture_frames = 0
            hand.is_waving = False
            return

        new_gesture = self.classify_gesture(hand.landmarks, handedness)

        # Track wave history (wrist x)
        hand.wave_history.append(hand.landmarks[0].x)
        if len(hand.wave_history) > 20:
            hand.wave_history.pop(0)
        hand.is_waving = self._detect_wave(hand)

        if new_gesture == hand.gesture:
            hand.gesture_frames += 1
        else:
            hand.prev_gesture = hand.gesture
            hand.gesture = new_gesture
            hand.gesture_frames = 0

    def _handle_thumbs_up(self):
        """Switch between drum and violin mode"""
        now = time.time()
        if now - self.last_thumbs_time < self.THUMBS_DEBOUNCE:
            return
        self.last_thumbs_time = now

        if self.mode == Mode.DRUMS:
            self.mode = Mode.VIOLIN
            if self.voice:
                import threading
                threading.Thread(target=self.voice.speak,
                                 args=("Switching to violin",), daemon=True).start()
        else:
            self.mode = Mode.DRUMS
            if self.voice:
                import threading
                threading.Thread(target=self.voice.speak,
                                 args=("Switching to drums",), daemon=True).start()

    def _fire_drum(self, sample_name: str, hand: HandState):
        """Trigger a drum hit with debounce"""
        now = time.time()
        if now - hand.last_trigger_time < self.TRIGGER_DEBOUNCE:
            return
        hand.last_trigger_time = now
        vol = 1.0 - self.right_hand.y_norm  # right hand height = volume
        vol = max(0.1, min(1.0, vol))
        self.sound.play(sample_name, volume=vol)

    def _update_violin(self):
        """Update violin note based on left hand height (continuous)"""
        if not self.left_hand.landmarks:
            return

        now = time.time()
        if now - self.last_violin_time < self.VIOLIN_UPDATE_INTERVAL:
            return
        self.last_violin_time = now

        # Mute on fist
        if self.left_hand.gesture == Gesture.FIST:
            self.sound.stop_loop("violin")
            return

        # Map Y position to pentatonic scale notes
        # y_norm: 0 = top of frame (high pitch), 1 = bottom (low pitch)
        y = self.left_hand.y_norm
        notes = ["violin_A3", "violin_C4", "violin_D4", "violin_E4",
                 "violin_G4", "violin_A4", "violin_C5", "violin_D5"]
        note_idx = int((1.0 - y) * (len(notes) - 1))
        note_idx = max(0, min(len(notes) - 1, note_idx))

        vol = 1.0 - self.right_hand.y_norm if self.right_hand.landmarks else 0.7
        vol = max(0.05, min(1.0, vol))

        # Vibrato from waving
        vibrato = self.left_hand.is_waving

        if note_idx != self.last_violin_note:
            self.last_violin_note = note_idx
            self.sound.play_loop("violin", notes[note_idx], volume=vol, vibrato=vibrato)
        else:
            self.sound.set_loop_volume("violin", vol)

    # ── ACTIONS PER MODE ─────────────────────────────────────────

    def _dispatch_drums(self):
        """Map current gestures to drum hits"""
        # Check both hands for thumbs up (mode switch)
        for hand, hn in [(self.left_hand, "Right"), (self.right_hand, "Left")]:
            if (hand.gesture == Gesture.THUMBS_UP and
                    hand.gesture_frames == self.GESTURE_CONFIRM_FRAMES):
                self._handle_thumbs_up()
                return

        # Drum mapping on LEFT hand (dominant rhythm hand)
        lh = self.left_hand
        if lh.gesture_frames == self.GESTURE_CONFIRM_FRAMES:
            if lh.gesture == Gesture.FIST:
                self._fire_drum("kick", lh)
            elif lh.gesture == Gesture.ONE_FINGER:
                self._fire_drum("snare", lh)
            elif lh.gesture == Gesture.TWO_FINGERS:
                self._fire_drum("hihat", lh)
            elif lh.gesture == Gesture.THREE_FINGERS:
                self._fire_drum("tom", lh)
            elif lh.gesture == Gesture.OPEN_PALM:
                self._fire_drum("crash", lh)
            elif lh.gesture == Gesture.PINCH:
                self._fire_drum("rimshot", lh)

        # Right hand: alternate hits for layering
        rh = self.right_hand
        if rh.gesture_frames == self.GESTURE_CONFIRM_FRAMES:
            if rh.gesture == Gesture.FIST:
                self._fire_drum("kick", rh)
            elif rh.gesture == Gesture.ONE_FINGER:
                self._fire_drum("hihat_open", rh)
            elif rh.gesture == Gesture.TWO_FINGERS:
                self._fire_drum("hihat", rh)
            elif rh.gesture == Gesture.OPEN_PALM:
                self._fire_drum("crash", rh)

    def _dispatch_violin(self):
        """Update violin continuous play"""
        # Check for mode switch
        for hand, hn in [(self.left_hand, "Right"), (self.right_hand, "Left")]:
            if (hand.gesture == Gesture.THUMBS_UP and
                    hand.gesture_frames == self.GESTURE_CONFIRM_FRAMES):
                self.sound.stop_loop("violin")
                self._handle_thumbs_up()
                return

        self._update_violin()

    # ── MAIN LOOP ────────────────────────────────────────────────

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror for natural feel
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self.hands.process(rgb)
            rgb.flags.writeable = True

            self._assign_hands(results)
            self._process_hand_gesture(self.left_hand, "Right", is_left=True)
            self._process_hand_gesture(self.right_hand, "Left", is_left=False)

            # Dispatch actions based on mode
            if self.mode == Mode.DRUMS:
                self._dispatch_drums()
            else:
                self._dispatch_violin()

            # Draw skeleton overlays
            if results.multi_hand_landmarks:
                for hand_lm, hand_info in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    label = hand_info.classification[0].label
                    color = (0, 255, 160) if label == "Right" else (255, 180, 0)
                    self._draw_hand(frame, hand_lm, color)

            # FPS counter
            self.frame_count += 1
            now = time.time()
            if now - self.fps_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.fps_time = now

            # Draw HUD overlay
            self.overlay.draw(frame, self)

            cv2.imshow("GESTURE.DJ", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def _draw_hand(self, frame, hand_lm, color):
        """Draw hand skeleton with custom color"""
        h, w = frame.shape[:2]
        lm = hand_lm.landmark

        # Draw connections
        connections = self.mp_hands.HAND_CONNECTIONS
        for conn in connections:
            p1 = lm[conn[0]]
            p2 = lm[conn[1]]
            x1, y1 = int(p1.x * w), int(p1.y * h)
            x2, y2 = int(p2.x * w), int(p2.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Draw landmark dots
        for i, point in enumerate(lm):
            x, y = int(point.x * w), int(point.y * h)
            # Fingertips slightly larger
            r = 5 if i in (4, 8, 12, 16, 20) else 3
            cv2.circle(frame, (x, y), r, color, -1, cv2.LINE_AA)
