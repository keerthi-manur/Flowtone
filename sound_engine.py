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
    # Drums use 8 channels (simultaneous hits), violin uses a dedicated channel
    DRUM_CHANNELS = 8
    VIOLIN_CHANNEL = 8  # channel index for looping violin
    TOTAL_CHANNELS = 9

    def __init__(self, samples_dir: str):
        self.samples_dir = samples_dir
        self._samples: Dict[str, pygame.mixer.Sound] = {}
        self._loop_channel: Optional[pygame.mixer.Channel] = None
        self._drum_channel_idx = 0
        self._drum_channels: list = []
        self._lock = threading.Lock()

        # Init pygame mixer with low latency settings
        pygame.mixer.pre_init(
            frequency=44100,
            size=-16,          # signed 16-bit
            channels=2,        # stereo
            buffer=256,        # small buffer = low latency (256 vs default 512)
        )
        pygame.mixer.init()
        pygame.mixer.set_num_channels(self.TOTAL_CHANNELS)

        # Allocate channels
        self._drum_channels = [pygame.mixer.Channel(i) for i in range(self.DRUM_CHANNELS)]
        self._loop_channel = pygame.mixer.Channel(self.VIOLIN_CHANNEL)

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

    # ── VIOLIN LOOPING ────────────────────────────────────────────

    def play_loop(self, instrument: str, sample_name: str,
                  volume: float = 0.7, vibrato: bool = False):
        """
        Play a sample in a loop on the dedicated loop channel.
        Crossfades when switching notes.
        """
        sample = self._get_sample(sample_name)
        if sample is None:
            return

        sample.set_volume(max(0.0, min(1.0, volume)))
        # Use loops=0 (play once) for long sustain samples to avoid loop cut artifacts
        # gesture_engine will retrigger as needed
        loops = 0 if sample_name.endswith("_long") else -1
        self._loop_channel.play(sample, loops=loops, fade_ms=120)

    def loop_is_busy(self) -> bool:
        return self._loop_channel.get_busy()

    def set_loop_volume(self, instrument: str, volume: float):
        """Update loop volume in real time (right hand height)"""
        if self._loop_channel.get_busy():
            # pygame doesn't have a direct channel volume setter for the
            # currently playing sound, so we set it on the channel itself
            self._loop_channel.set_volume(max(0.0, min(1.0, volume)))

    def stop_loop(self, instrument: str):
        """Stop the looping violin note (fade out)"""
        self._loop_channel.fadeout(80)

    # ── CLEANUP ───────────────────────────────────────────────────

    def shutdown(self):
        pygame.mixer.fadeout(200)
        time.sleep(0.3)
        pygame.mixer.quit()