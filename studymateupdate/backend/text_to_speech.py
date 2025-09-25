import pyttsx3
import os
from typing import Optional

def init_engine(rate: int = 170, volume: float = 1.0, voice_index: Optional[int] = None):
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)
    voices = engine.getProperty("voices")
    if voice_index is not None and 0 <= voice_index < len(voices):
        engine.setProperty("voice", voices[voice_index].id)
    return engine

def save_speech(text: str, filename: str) -> str:
    try:
        engine = init_engine()
        if os.path.exists(filename):
            os.remove(filename)
        engine.save_to_file(text, filename)
        engine.runAndWait()
        engine.stop()
        return filename
    except Exception as e:
        raise RuntimeError(f"save_speech error: {e}")

def speak_text(text: str):
    try:
        engine = init_engine()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        raise RuntimeError(f"speak_text error: {e}")
