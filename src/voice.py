"""
Voice I/O module.

STT: faster-whisper (local, CPU)
TTS: pyttsx3 (local, espeak on Linux)

Both components fail gracefully — callers receive None / empty string and
should fall back to text-only mode.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger("wellness_companion.voice")

# ── Speech-to-Text ──────────────────────────────────────────────────────────

def transcribe(audio_path: str | Path) -> Optional[str]:
    """
    Transcribe an audio file to text using faster-whisper.
    Forces English language.
    Returns None on failure.
    """
    try:
        from faster_whisper import WhisperModel
        from src.config import WHISPER_MODEL

        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        segments, info = model.transcribe(str(audio_path), beam_size=5, language="en")
        text = " ".join(seg.text.strip() for seg in segments).strip()
        logger.info("STT transcribed (lang=%s): %r", info.language, text[:80])
        return text if text else None
    except Exception as exc:
        logger.warning("STT failed: %s", exc)
        return None


# ── Text-to-Speech ──────────────────────────────────────────────────────────

def speak(text: str) -> bool:
    """
    Speak text aloud using pyttsx3.
    Returns True on success, False on failure.
    """
    # Strip markdown formatting before speaking
    import re
    clean = re.sub(r"\*+", "", text)
    clean = re.sub(r"#+\s", "", clean)
    clean = re.sub(r"\n+", " ", clean).strip()

    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", 160)   # words per minute
        engine.setProperty("volume", 0.9)
        engine.say(clean)
        engine.runAndWait()
        return True
    except Exception as exc:
        logger.warning("TTS failed: %s", exc)
        return False


def tts_to_file(text: str) -> Optional[Path]:
    """
    Synthesise speech and save to a temp WAV file.
    Returns path on success, None on failure (Streamlit can play the file).
    """
    import re
    import time
    
    clean = re.sub(r"\*+", "", text)
    clean = re.sub(r"#+\s", "", clean)
    clean = re.sub(r"\n+", " ", clean).strip()

    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.setProperty("volume", 0.9)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_name = tmp.name
        tmp.close()
        
        # Save to file and wait for completion
        engine.save_to_file(clean, tmp_name)
        engine.runAndWait()
        
        # Give the file system time to flush
        time.sleep(0.5)
        
        path = Path(tmp_name)
        if path.exists() and path.stat().st_size > 1000:  # At least 1KB of audio data
            logger.info("TTS saved to %s (%d bytes)", tmp_name, path.stat().st_size)
            return path
        else:
            logger.warning("TTS file not created or too small (%d bytes)", path.stat().st_size if path.exists() else 0)
            return None
    except Exception as exc:
        logger.warning("TTS to file failed: %s", exc)
        return None
