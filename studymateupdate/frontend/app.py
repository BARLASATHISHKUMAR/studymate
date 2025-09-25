import os
import sys
import tempfile
import gradio as gr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.pdf_handler import extract_text_from_pdf
from backend.text_analysis import analyze_text
from backend.speech_to_text import speech_to_text
from backend.text_to_speech import save_speech
from backend.text_chunker import chunk_text
from backend.embeddings_index import build_faiss_index, search_faiss_index

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def process_pdf(file):
    try:
        save_path = os.path.join(UPLOAD_FOLDER, file.name)
        with open(save_path, "wb") as f:
            f.write(file.read())
        text = extract_text_from_pdf(save_path)
        return text if text.strip() else "⚠ No extractable text found."
    except Exception as e:
        return f"❌ Error reading PDF: {e}"

state = gr.State({"pdf_text": "", "chunks": [], "faiss_index": None, "embeddings": None})

def upload_pdf(files, state):
    if not files:
        return "No files uploaded.", state
    texts = [process_pdf(f) for f in files]
    combined_text = "\n\n".join([t for t in texts if t])
    state["pdf_text"] = combined_text
    # Chunk and build FAISS
    chunks = chunk_text(combined_text)
    state["chunks"] = chunks
    index, embeddings = build_faiss_index(chunks)
    state["faiss_index"] = index
    state["embeddings"] = embeddings
    return "PDFs processed and indexed successfully.", state

def text_question(input_text, state):
    if not state.get("pdf_text"):
        return "Please upload PDFs first.", None, state
    try:
        context = search_faiss_index(input_text, state["faiss_index"], state["chunks"], state["embeddings"])
        answer = analyze_text(input_text, context)
        temp_audio = os.path.join(tempfile.gettempdir(), "studymate_answer.wav")
        save_speech(answer, temp_audio)
        return answer, temp_audio, state
    except Exception as e:
        return f"Error generating answer: {e}", None, state

def speech_question(audio_file, state):
    try:
        query = speech_to_text(audio_file)
        return text_question(query, state)
    except Exception as e:
        return f"Error in speech question: {e}", None, state

with gr.Blocks() as demo:
    gr.Markdown("# 📘 StudyMate — AI Academic Assistant")
    state_obj = gr.State({"pdf_text": "", "chunks": [], "faiss_index": None, "embeddings": None})

    with gr.Row():
        pdf_upload = gr.File(file_count="multiple", label="Upload PDFs")
        upload_btn = gr.Button("Process PDFs")
    status = gr.Textbox(label="Status")
    upload_btn.click(upload_pdf, inputs=[pdf_upload, state_obj], outputs=[status, state_obj])

    with gr.Row():
        question = gr.Textbox(label="Ask a question", placeholder="Type your question here...")
        ask_btn = gr.Button("Ask")
    answer = gr.Textbox(label="Answer")
    audio = gr.Audio(label="Answer (TTS)")
    ask_btn.click(fn=text_question, inputs=[question, state_obj], outputs=[answer, audio, state_obj])

    with gr.Row():
        speech_btn = gr.Button("Ask by Speaking")
    speech_answer = gr.Textbox(label="Answer from Speech")
    speech_audio = gr.Audio(label="Answer (TTS)")
    speech_btn.click(fn=speech_question, inputs=[gr.Audio(source="microphone", type="filepath"), state_obj],
                     outputs=[speech_answer, speech_audio, state_obj])

if __name__ == "__main__":
    demo.launch(share=False)