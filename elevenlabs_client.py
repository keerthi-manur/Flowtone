"""
elevenlabs_client.py
Handles ElevenLabs TTS: pre-generates audio at startup,
plays instantly during performance (no API latency mid-show).
"""

import os
import io
import time
import threading
import requests
import pygame
from typing import Dict, Optional


class ElevenLabsClient:
    # Rachel voice — warm, clear, musical feel
    DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
    API_URL = "https://api.elevenlabs.io/v1/text-to-speech"
    MODEL = "eleven_turbo_v2"  # Fastest model, good for short phrases

    def __init__(self, api_key: str, voice_id: Optional[str] = None):
        self.api_key = api_key
        self.voice_id = voice_id or self.DEFAULT_VOICE_ID
        self._cache: Dict[str, bytes] = {}  # phrase → raw mp3 bytes
        self._lock = threading.Lock()

        # Ensure pygame mixer is up (sound_engine inits it, but just in case)
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

    def _generate(self, text: str) -> Optional[bytes]:
        """Call ElevenLabs API and return mp3 bytes"""
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": self.MODEL,
            "voice_settings": {
                "stability": 0.4,
                "similarity_boost": 0.8,
                "style": 0.2,
                "use_speaker_boost": True,
            }
        }
        url = f"{self.API_URL}/{self.voice_id}"
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=10)
            r.raise_for_status()
            return r.content
        except Exception as e:
            print(f"   ⚠️  ElevenLabs error for '{text}': {e}")
            return None

    def prewarm(self, phrases: list[str]):
        """
        Pre-generate all phrases in parallel threads at startup.
        This way speak() is instant during performance.
        """
        def _fetch(phrase):
            audio = self._generate(phrase)
            if audio:
                with self._lock:
                    self._cache[phrase] = audio
                print(f"   ✓ voice: '{phrase}'")

        threads = [threading.Thread(target=_fetch, args=(p,), daemon=True)
                   for p in phrases]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

    def speak(self, text: str):
        """
        Play a phrase. Uses cache if available, otherwise generates on the fly.
        Called from background thread so it never blocks the gesture loop.
        """
        with self._lock:
            audio_bytes = self._cache.get(text)

        if audio_bytes is None:
            # Not pre-warmed — generate now (will have latency)
            audio_bytes = self._generate(text)
            if audio_bytes is None:
                return
            with self._lock:
                self._cache[text] = audio_bytes

        try:
            sound = pygame.mixer.Sound(io.BytesIO(audio_bytes))
            # Use channel 9 for voice (won't stomp on music)
            voice_ch = pygame.mixer.Channel(
                min(9, pygame.mixer.get_num_channels() - 1)
            )
            voice_ch.play(sound)
        except Exception as e:
            print(f"   ⚠️  Voice playback error: {e}")
