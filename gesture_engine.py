"""
gesture_engine.py
MediaPipe hand tracking → gesture classification → sound/voice triggers
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Mode(Enum):
    DRUMS  = "drums"
    VIOLIN = "violin"


class Gesture(Enum):
    FIST          = "fist"
    ONE_FINGER    = "one_finger"
    TWO_FINGERS   = "two_fingers"
    THREE_FINGERS = "three_fingers"
    OPEN_PALM     = "open_palm"
    THUMBS_UP     = "thumbs_up"
    WAVE          = "wave"
    PINCH         = "pinch"
    NEUTRAL       = "neutral"


@dataclass
class HandState:
    landmarks: list = field(default_factory=list)
    gesture: Gesture = Gesture.NEUTRAL
    prev_gesture: Gesture = Gesture.NEUTRAL
    gesture_frames: int = 0
    y_norm: float = 0.5
    x_norm: float = 0.5
    wave_history: list = field(default_factory=list)
    is_waving: bool = False
    last_trigger_time: float = 0.0


class GestureEngine:
    GESTURE_CONFIRM_FRAMES = 4
    TRIGGER_DEBOUNCE       = 0.22
    VIOLIN_UPDATE_INTERVAL = 0.05
    THUMBS_DEBOUNCE        = 1.2
    WAVE_WINDOW            = 14
    WAVE_THRESHOLD         = 0.06

    def __init__(self, camera_index, sound, voice, overlay, mirror=False):
        self.sound   = sound
        self.voice   = voice
        self.overlay = overlay
        self.mirror  = mirror
        self.mode    = Mode.DRUMS

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.72,
            min_tracking_confidence=0.65,
        )

        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.left_hand  = HandState()
        self.right_hand = HandState()

        self.last_thumbs_time = 0.0
        self.last_violin_time = 0.0
        self.last_violin_note = -1

        self.frame_count = 0
        self.fps         = 0
        self.fps_time    = time.time()

        self.left_label  = "—"
        self.right_label = "—"

    # ── LANDMARK HELPERS ─────────────────────────────────────────

    def _finger_up(self, lm, tip, pip):
        return lm[tip].y < lm[pip].y - 0.02

    def _count_fingers(self, lm):
        return sum(self._finger_up(lm, t, p)
                   for t, p in [(8,6),(12,10),(16,14),(20,18)])

    def _thumb_up_gesture(self, lm):
        """
        Reliable thumbs-up detection:
        - Thumb tip is well above the wrist
        - Thumb tip is higher than all other fingertips
        - All 4 fingers are curled (tip below its PIP joint, not just MCP)
        - Thumb tip is above the index MCP (knuckle) — rules out fist
        """
        thumb_tip = lm[4]
        wrist     = lm[0]

        # Thumb tip must be clearly above wrist
        if thumb_tip.y > wrist.y - 0.15:
            return False

        # Thumb tip must be above the index finger knuckle (rules out fist)
        if thumb_tip.y > lm[5].y:
            return False

        # Thumb tip must be above all other fingertips
        other_tips = [lm[i] for i in (8, 12, 16, 20)]
        if any(thumb_tip.y > tip.y for tip in other_tips):
            return False

        # ALL fingers must be firmly curled: tip below PIP joint (stricter than MCP)
        curled = sum(lm[tip].y > lm[pip].y
                     for tip, pip in [(8,6),(12,10),(16,14),(20,18)])
        return curled == 4

    def _pinch_dist(self, lm):
        return (((lm[4].x-lm[8].x)**2 + (lm[4].y-lm[8].y)**2) ** 0.5)

    def classify_gesture(self, lm) -> Gesture:
        if self._thumb_up_gesture(lm):
            return Gesture.THUMBS_UP

        n = self._count_fingers(lm)

        if self._pinch_dist(lm) < 0.05 and n <= 1:
            return Gesture.PINCH
        if n == 0: return Gesture.FIST
        if n == 1: return Gesture.ONE_FINGER
        if n == 2: return Gesture.TWO_FINGERS
        if n == 3: return Gesture.THREE_FINGERS
        return Gesture.OPEN_PALM

    def _detect_wave(self, hand: HandState) -> bool:
        if len(hand.wave_history) < self.WAVE_WINDOW:
            return False
        h = hand.wave_history[-self.WAVE_WINDOW:]
        changes = 0
        for i in range(1, len(h)-1):
            if abs(h[i] - h[i-1]) > self.WAVE_THRESHOLD:
                if (h[i]-h[i-1] > 0) != (h[i+1]-h[i] > 0):
                    changes += 1
        return changes >= 2

    # ── HAND ASSIGNMENT ──────────────────────────────────────────

    def _assign_hands(self, results):
        self.left_hand.landmarks  = []
        self.right_hand.landmarks = []

        if not results.multi_hand_landmarks:
            return

        for hand_lm, hand_info in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            lm    = hand_lm.landmark
            label = hand_info.classification[0].label

            # After cv2.flip(frame,1), MediaPipe "Right" = user's LEFT hand.
            # --mirror inverts this for cameras that are already flipped.
            is_users_left = (label == "Right") if not self.mirror else (label == "Left")

            if is_users_left:
                self.left_hand.landmarks = lm
                self.left_hand.x_norm   = lm[0].x
                self.left_hand.y_norm   = lm[0].y
            else:
                self.right_hand.landmarks = lm
                self.right_hand.x_norm   = lm[0].x
                self.right_hand.y_norm   = lm[0].y

    # ── GESTURE STATE UPDATE ─────────────────────────────────────

    def _update_hand(self, hand: HandState):
        if not hand.landmarks:
            hand.gesture        = Gesture.NEUTRAL
            hand.gesture_frames = 0
            hand.is_waving      = False
            return

        new_g = self.classify_gesture(hand.landmarks)

        hand.wave_history.append(hand.landmarks[0].x)
        if len(hand.wave_history) > 20:
            hand.wave_history.pop(0)
        hand.is_waving = self._detect_wave(hand)

        if new_g == hand.gesture:
            hand.gesture_frames += 1
        else:
            hand.prev_gesture   = hand.gesture
            hand.gesture        = new_g
            hand.gesture_frames = 0

    # ── MODE SWITCH ──────────────────────────────────────────────

    def _handle_thumbs_up(self):
        now = time.time()
        if now - self.last_thumbs_time < self.THUMBS_DEBOUNCE:
            return
        self.last_thumbs_time = now
        self.sound.stop_loop("violin")

        if self.mode == Mode.DRUMS:
            self.mode  = Mode.VIOLIN
            phrase     = "Switching to violin"
        else:
            self.mode  = Mode.DRUMS
            phrase     = "Switching to drums"

        if self.voice:
            threading.Thread(target=self.voice.speak,
                             args=(phrase,), daemon=True).start()

    # ── DRUM DISPATCH ────────────────────────────────────────────

    def _fire(self, sample, hand: HandState):
        now = time.time()
        if now - hand.last_trigger_time < self.TRIGGER_DEBOUNCE:
            return
        hand.last_trigger_time = now
        vol = max(0.15, min(1.0, 1.0 - self.right_hand.y_norm))
        self.sound.play(sample, volume=vol)

    def _dispatch_drums(self):
        for hand in (self.left_hand, self.right_hand):
            if (hand.gesture == Gesture.THUMBS_UP and
                    hand.gesture_frames == self.GESTURE_CONFIRM_FRAMES):
                self._handle_thumbs_up()
                return

        lh = self.left_hand
        # Fire once gesture is confirmed, then again whenever debounce clears
        if lh.gesture_frames >= self.GESTURE_CONFIRM_FRAMES:
            hit = {
                Gesture.FIST:          "kick",
                Gesture.ONE_FINGER:    "snare",
                Gesture.TWO_FINGERS:   "hihat",
                Gesture.THREE_FINGERS: "tom",
                Gesture.OPEN_PALM:     "crash",
                Gesture.PINCH:         "rimshot",
            }.get(lh.gesture)
            if hit:
                self._fire(hit, lh)

        rh = self.right_hand
        if rh.gesture_frames >= self.GESTURE_CONFIRM_FRAMES:
            hit = {
                Gesture.FIST:        "kick",
                Gesture.ONE_FINGER:  "hihat_open",
                Gesture.TWO_FINGERS: "hihat",
                Gesture.OPEN_PALM:   "crash",
            }.get(rh.gesture)
            if hit:
                self._fire(hit, rh)

    # ── VIOLIN DISPATCH ──────────────────────────────────────────

    def _dispatch_violin(self):
        for hand in (self.left_hand, self.right_hand):
            if (hand.gesture == Gesture.THUMBS_UP and
                    hand.gesture_frames == self.GESTURE_CONFIRM_FRAMES):
                self._handle_thumbs_up()
                return

        if self.left_hand.gesture == Gesture.FIST:
            self.sound.stop_loop("violin")
            return

        now = time.time()
        if now - self.last_violin_time < self.VIOLIN_UPDATE_INTERVAL:
            return
        self.last_violin_time = now

        if not self.left_hand.landmarks:
            return

        notes = ["violin_A3","violin_C4","violin_D4","violin_E4",
                 "violin_G4","violin_A4","violin_C5","violin_D5"]
        idx  = int((1.0 - self.left_hand.y_norm) * (len(notes)-1))
        idx  = max(0, min(len(notes)-1, idx))
        vol  = max(0.05, min(1.0,
               1.0 - self.right_hand.y_norm if self.right_hand.landmarks else 0.7))

        if idx != self.last_violin_note:
            self.last_violin_note = idx
            self.sound.play_loop("violin", notes[idx], volume=vol,
                                 vibrato=self.left_hand.is_waving)
        else:
            self.sound.set_loop_volume("violin", vol)

    # ── LABELS FOR HUD ───────────────────────────────────────────

    DRUM_MAP = {
        Gesture.FIST:          "KICK",
        Gesture.ONE_FINGER:    "SNARE",
        Gesture.TWO_FINGERS:   "HI-HAT",
        Gesture.THREE_FINGERS: "TOM",
        Gesture.OPEN_PALM:     "CRASH",
        Gesture.PINCH:         "RIMSHOT",
        Gesture.THUMBS_UP:     "SWITCH →",
        Gesture.NEUTRAL:       "—",
        Gesture.WAVE:          "—",
    }
    VIOLIN_MAP = {
        Gesture.FIST:          "MUTE",
        Gesture.OPEN_PALM:     "BOW",
        Gesture.WAVE:          "VIBRATO",
        Gesture.THUMBS_UP:     "SWITCH →",
        Gesture.NEUTRAL:       "BOW",
        Gesture.ONE_FINGER:    "BOW",
        Gesture.TWO_FINGERS:   "BOW",
        Gesture.THREE_FINGERS: "BOW",
        Gesture.PINCH:         "MUTE",
    }

    def _gesture_label(self, hand: HandState) -> str:
        m = self.DRUM_MAP if self.mode == Mode.DRUMS else self.VIOLIN_MAP
        if hand.is_waving and self.mode == Mode.VIOLIN:
            return "VIBRATO"
        return m.get(hand.gesture, "—")

    # ── MAIN LOOP ────────────────────────────────────────────────

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self.hands.process(rgb)
            rgb.flags.writeable = True

            self._assign_hands(results)
            self._update_hand(self.left_hand)
            self._update_hand(self.right_hand)

            if self.mode == Mode.DRUMS:
                self._dispatch_drums()
            else:
                self._dispatch_violin()

            self.left_label  = self._gesture_label(self.left_hand)
            self.right_label = self._gesture_label(self.right_hand)

            if results.multi_hand_landmarks:
                for hand_lm, hand_info in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):
                    label     = hand_info.classification[0].label
                    is_left   = (label == "Right") if not self.mirror else (label == "Left")
                    color     = (0, 255, 160) if is_left else (255, 180, 0)
                    self._draw_hand(frame, hand_lm, color)

            self.frame_count += 1
            now = time.time()
            if now - self.fps_time >= 1.0:
                self.fps        = self.frame_count
                self.frame_count = 0
                self.fps_time   = now

            self.overlay.draw(frame, self)
            cv2.imshow("GESTURE.DJ", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def _draw_hand(self, frame, hand_lm, color):
        h, w = frame.shape[:2]
        lm   = hand_lm.landmark
        for conn in self.mp_hands.HAND_CONNECTIONS:
            p1 = lm[conn[0]]; p2 = lm[conn[1]]
            cv2.line(frame,
                     (int(p1.x*w), int(p1.y*h)),
                     (int(p2.x*w), int(p2.y*h)),
                     color, 2, cv2.LINE_AA)
        for i, pt in enumerate(lm):
            r = 5 if i in (4,8,12,16,20) else 3
            cv2.circle(frame, (int(pt.x*w), int(pt.y*h)),
                       r, color, -1, cv2.LINE_AA)