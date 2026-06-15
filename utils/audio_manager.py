"""
Audio manager for the Driver Drowsiness Detection System.

Generates/loads alarm audio and renders a browser-native HTML/JS audio
component so the alarm works both locally and on Streamlit Cloud without
any native audio libraries.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import os
import struct
import wave
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WAV synthesis helper
# ---------------------------------------------------------------------------

def _generate_beep_wav(
    frequency: float = 880.0,
    duration_ms: int = 800,
    sample_rate: int = 44100,
    amplitude: float = 0.6,
) -> bytes:
    """
    Synthesise a pure-tone beep as raw WAV bytes.
    Returns a bytes object containing a valid .wav file.
    """
    n_samples = int(sample_rate * duration_ms / 1000)
    attack = int(sample_rate * 0.01)   # 10 ms attack
    release = int(sample_rate * 0.05)  # 50 ms release

    samples: list[int] = []
    for i in range(n_samples):
        t = i / sample_rate
        value = amplitude * math.sin(2 * math.pi * frequency * t)

        # Apply simple envelope
        if i < attack:
            value *= i / attack
        elif i > n_samples - release:
            value *= (n_samples - i) / release

        samples.append(int(value * 32767))

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)                 # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))
    return buf.getvalue()


def _generate_alarm_wav() -> bytes:
    """
    Build a more attention-grabbing alarm by stacking alternating tones.
    Returns a mono 44100 Hz 16-bit WAV.
    """
    sample_rate = 44100
    amplitude = 0.65

    # Two-tone siren: 880 Hz → 660 Hz repeating
    cycles = 3
    tone_ms = 400
    n_per_tone = int(sample_rate * tone_ms / 1000)
    freqs = [880.0, 660.0]

    all_samples: list[int] = []
    for _ in range(cycles):
        for freq in freqs:
            attack = int(sample_rate * 0.008)
            release = int(sample_rate * 0.04)
            for i in range(n_per_tone):
                t = i / sample_rate
                v = amplitude * math.sin(2 * math.pi * freq * t)
                if i < attack:
                    v *= i / attack
                elif i > n_per_tone - release:
                    v *= (n_per_tone - i) / release
                all_samples.append(int(v * 32767))

    buf = io.BytesIO()
    n = len(all_samples)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *all_samples))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Audio manager class
# ---------------------------------------------------------------------------

class AudioManager:
    """
    Manages alarm audio for the drowsiness detection app.

    Audio is delivered entirely via a browser-native HTML <audio> element
    controlled by inline JavaScript — no pygame / pyaudio needed, so it
    works on Streamlit Cloud.
    """

    DEFAULT_ALARM_PATH = "assets/default_alarm.wav"

    def __init__(self) -> None:
        self._alarm_b64: Optional[str] = None
        self._mime: str = "audio/wav"
        self._ensure_default_alarm()

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def _ensure_default_alarm(self) -> None:
        """Create the default alarm WAV if it does not exist."""
        os.makedirs("assets", exist_ok=True)
        if not os.path.exists(self.DEFAULT_ALARM_PATH):
            wav_bytes = _generate_alarm_wav()
            with open(self.DEFAULT_ALARM_PATH, "wb") as f:
                f.write(wav_bytes)
            logger.info("Created default alarm at %s", self.DEFAULT_ALARM_PATH)
        self.load_from_file(self.DEFAULT_ALARM_PATH)

    def load_from_file(self, path: str) -> None:
        """Load alarm audio from a file path and base64-encode it."""
        try:
            with open(path, "rb") as f:
                raw = f.read()
            self._alarm_b64 = base64.b64encode(raw).decode()
            if path.lower().endswith(".mp3"):
                self._mime = "audio/mpeg"
            elif path.lower().endswith(".ogg"):
                self._mime = "audio/ogg"
            else:
                self._mime = "audio/wav"
            logger.info("Loaded alarm audio from %s (%d bytes)", path, len(raw))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load alarm file %s: %s", path, exc)

    def load_from_bytes(self, data: bytes, mime: str = "audio/wav") -> None:
        """Load alarm audio from raw bytes (e.g. Streamlit file_uploader)."""
        self._alarm_b64 = base64.b64encode(data).decode()
        self._mime = mime
        logger.info("Loaded alarm from bytes (%d bytes, %s)", len(data), mime)

    # ------------------------------------------------------------------ #
    # Streamlit rendering                                                  #
    # ------------------------------------------------------------------ #

    def render_alarm_component(
        self,
        playing: bool,
        volume: float = 0.8,
        component_key: str = "alarm_audio",
    ) -> None:
        """
        Inject an invisible HTML <audio> element controlled by JS.

        Call this once per Streamlit rerun; set `playing=True` to start
        the alarm loop and `playing=False` to stop it.
        """
        if self._alarm_b64 is None:
            return

        play_js = "true" if playing else "false"
        vol = max(0.0, min(1.0, volume))

        html = f"""
        <audio id="ddd-alarm" preload="auto" loop>
            <source src="data:{self._mime};base64,{self._alarm_b64}" type="{self._mime}">
        </audio>
        <script>
        (function() {{
            var audio = document.getElementById('ddd-alarm');
            if (!audio) return;
            audio.volume = {vol:.2f};
            var shouldPlay = {play_js};
            if (shouldPlay) {{
                if (audio.paused) {{
                    audio.play().catch(function(e) {{
                        console.warn('Autoplay blocked: ' + e);
                    }});
                }}
            }} else {{
                if (!audio.paused) {{
                    audio.pause();
                    audio.currentTime = 0;
                }}
            }}
        }})();
        </script>
        """
        components.html(html, height=0)

    def render_test_button_component(self, volume: float = 0.8) -> None:
        """Render a one-shot play button for testing the alarm sound."""
        if self._alarm_b64 is None:
            st.warning("No alarm audio loaded.")
            return

        vol = max(0.0, min(1.0, volume))
        html = f"""
        <audio id="ddd-test-alarm" preload="auto">
            <source src="data:{self._mime};base64,{self._alarm_b64}" type="{self._mime}">
        </audio>
        <button
            onclick="
                var a = document.getElementById('ddd-test-alarm');
                a.volume = {vol:.2f};
                a.currentTime = 0;
                a.play();
            "
            style="
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                border: none;
                padding: 8px 20px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 14px;
                font-family: Inter, sans-serif;
                font-weight: 600;
                letter-spacing: 0.5px;
            "
        >
            🔔 Test Alarm
        </button>
        """
        components.html(html, height=50)

    @staticmethod
    def get_audio_data_uri(path: str) -> Optional[str]:
        """Return a data URI for an audio file (for st.audio preview)."""
        try:
            with open(path, "rb") as f:
                raw = f.read()
            b64 = base64.b64encode(raw).decode()
            mime = "audio/wav"
            if path.endswith(".mp3"):
                mime = "audio/mpeg"
            return f"data:{mime};base64,{b64}"
        except Exception:  # noqa: BLE001
            return None
