from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def build_faiss_index(text_chunks):
    embeddings = model.encode(text_chunks, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def search_faiss_index(query, index, text_chunks, embeddings, top_k=3):
    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, top_k)
    return "\n\n".join([text_chunks[i] for i in I[0]])
