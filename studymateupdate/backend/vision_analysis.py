from dotenv import load_dotenv
import os
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, AutoProcessor, AutoTokenizer

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MODEL_NAME = os.getenv("GRANITE_VISION_MODEL", "ibm-granite/granite-vision-3.2-2b")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME, token=HUGGINGFACE_TOKEN)
processor = AutoProcessor.from_pretrained(MODEL_NAME, token=HUGGINGFACE_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HUGGINGFACE_TOKEN)
model.to(device)

def analyze_image_with_text(image_source, prompt: str = "Describe the image:") -> str:
    try:
        image = Image.open(image_source).convert("RGB")
        if max(image.size) > 1024:
            image.thumbnail((1024, 1024))
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        decoder_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        generated_ids = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=200,
            temperature=0.2,
            top_p=0.9,
            do_sample=False
        )
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if prompt.strip() in text:
            text = text.split(prompt.strip())[-1].strip()
        return text
    except Exception as e:
        raise RuntimeError(f"vision_analysis error: {e}")
