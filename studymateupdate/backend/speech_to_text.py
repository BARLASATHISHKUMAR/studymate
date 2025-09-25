from dotenv import load_dotenv
import os
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MODEL_NAME = os.getenv("WHISPER_MODEL", "openai/whisper-medium")

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = WhisperProcessor.from_pretrained(MODEL_NAME, token=HUGGINGFACE_TOKEN)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, token=HUGGINGFACE_TOKEN)
model.to(device)

def speech_to_text(audio_path: str) -> str:
    try:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        generated_ids = model.generate(input_features)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return transcription
    except Exception as e:
        raise RuntimeError(f"speech_to_text error: {e}")
