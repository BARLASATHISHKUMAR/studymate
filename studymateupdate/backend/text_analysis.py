from dotenv import load_dotenv
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MODEL_NAME = os.getenv("GRANITE_MODEL", "ibm-granite/granite-3.2-2b-instruct")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HUGGINGFACE_TOKEN, use_slow_tokenizer=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HUGGINGFACE_TOKEN)
model.to(device)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

def analyze_text(query: str, context: str = "") -> str:
    try:
        prompt = f"You are StudyMate, an academic assistant.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.2,
            top_p=0.95,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in generated:
            generated = generated.split("Answer:")[-1].strip()
        return generated
    except Exception as e:
        raise RuntimeError(f"analyze_text error: {e}")
