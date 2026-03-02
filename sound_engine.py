"""
sound_engine.py
Handles all audio: pre-loaded samples, drum hits, looping violin notes.
Uses pygame.mixer for low-latency playback.
"""

import pygame
import os
import time
import threading
from typing import Dict, Optional


class SoundEngine:
    DRUM_CHANNELS   = 8
    VIOLIN_CHANNEL  = 8
    FLUTE_L_CHANNEL = 9
    FLUTE_R_CHANNEL = 10
    TOTAL_CHANNELS  = 11

    def __init__(self, samples_dir: str):
        self.samples_dir = samples_dir
        self._samples: Dict[str, pygame.mixer.Sound] = {}
        self._drum_channel_idx = 0
        self._drum_channels: list = []
        self._lock = threading.Lock()

        pygame.mixer.pre_init(
            frequency=44100,
            size=-16,
            channels=2,
            buffer=256,
        )
        pygame.mixer.init()
        pygame.mixer.set_num_channels(self.TOTAL_CHANNELS)

        self._drum_channels = [pygame.mixer.Channel(i) for i in range(self.DRUM_CHANNELS)]
        self._loop_channels = {
            "violin":      pygame.mixer.Channel(self.VIOLIN_CHANNEL),
            "flute_left":  pygame.mixer.Channel(self.FLUTE_L_CHANNEL),
            "flute_right": pygame.mixer.Channel(self.FLUTE_R_CHANNEL),
        }

    # ── SAMPLE LOADING ────────────────────────────────────────────

    def preload_all(self):
        """Pre-load all .wav files from samples_dir into memory"""
        if not os.path.isdir(self.samples_dir):
            print(f"   ⚠️  Samples directory '{self.samples_dir}' not found.")
            print(f"   ℹ️  Run: python download_samples.py  to download free samples.")
            print(f"   ℹ️  Or place your own .wav files in ./{self.samples_dir}/\n")
            return

        for fname in os.listdir(self.samples_dir):
            if fname.endswith(".wav") or fname.endswith(".mp3"):
                key = fname.rsplit(".", 1)[0]  # strip extension regardless of type
                path = os.path.join(self.samples_dir, fname)
                try:
                    self._samples[key] = pygame.mixer.Sound(path)
                    print(f"   ✓ {key}")
                except Exception as e:
                    print(f"   ✗ {key}: {e}")

    def sample_count(self) -> int:
        return len(self._samples)

    def has_sample(self, name: str) -> bool:
        return name in self._samples

    def _get_sample(self, name: str) -> Optional[pygame.mixer.Sound]:
        """Return sample or None with a helpful warning"""
        s = self._samples.get(name)
        if s is None and self._samples:
            # Try a fallback — any sample with a partial name match
            for k in self._samples:
                if name.split("_")[0] in k:
                    return self._samples[k]
        return s

    # ── DRUM PLAYBACK ─────────────────────────────────────────────

    def play(self, name: str, volume: float = 1.0):
        """Play a one-shot sample (non-blocking, round-robin channels)"""
        sample = self._get_sample(name)
        if sample is None:
            return

        sample.set_volume(max(0.0, min(1.0, volume)))

        with self._lock:
            ch = self._drum_channels[self._drum_channel_idx % self.DRUM_CHANNELS]
            self._drum_channel_idx += 1

        ch.play(sample)

    # ── LOOPING ───────────────────────────────────────────────────

    def play_loop(self, instrument: str, sample_name: str,
                  volume: float = 0.7, vibrato: bool = False):
        sample = self._get_sample(sample_name)
        if sample is None:
            return
        ch = self._loop_channels.get(instrument)
        if ch is None:
            return
        loops = 0 if (sample_name.endswith("_long") or "trill" in sample_name) else -1
        sample.set_volume(max(0.0, min(1.0, volume)))
        ch.play(sample, loops=loops, fade_ms=120)

    def loop_is_busy(self, instrument: str = "violin") -> bool:
        ch = self._loop_channels.get(instrument)
        return ch.get_busy() if ch else False

    def set_loop_volume(self, instrument: str, volume: float):
        ch = self._loop_channels.get(instrument)
        if ch and ch.get_busy():
            ch.set_volume(max(0.0, min(1.0, volume)))

    def stop_loop(self, instrument: str):
        ch = self._loop_channels.get(instrument)
        if ch:
            ch.fadeout(80)

    # ── CLEANUP ───────────────────────────────────────────────────

    def shutdown(self):
        pygame.mixer.fadeout(200)
        time.sleep(0.3)
        pygame.mixer.quit()