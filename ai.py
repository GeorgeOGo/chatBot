import os
import json
import getpass
import faiss
import langdetect
from langchain.chat_models import init_chat_model
from sentence_transformers import SentenceTransformer
from docx import Document
import numpy as np
from flask import Flask, request, jsonify

# Set up Groq API Key
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")
# 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

app = Flask(__name__)

# Your existing functions remain unchanged
def load_docx(filepath):
    doc = Document(filepath)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def chunk_text(text, chunk_size=512):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def create_faiss_index(text_chunks, embedding_model):
    embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, text_chunks

def retrieve_top_k(query, index, text_chunks, embedding_model, k=3):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    retrieved_texts = [text_chunks[i] for i in indices[0] if i < len(text_chunks)]
    if not retrieved_texts:
        return "لم أتمكن من العثور على إجابة لسؤالك، يمكنك التواصل مع فريق الدعم عبر support@youlearnt.com."
    return " ".join(retrieved_texts)

def generate_response(model, query, context, language):
    prompt = f"""
    Context: {context}
    Question: {query}
    Answer (respond in {language}):
    """
    response = model.invoke(prompt)
    if not response:
        return "لم أتمكن من العثور على إجابة لسؤالك." if language == 'ar' else "I couldn't find an answer to your question."
    return response.content

def process_query(doc_path, user_query):
    # Detect language
    language = langdetect.detect(user_query)
    lang_map = {'ar': 'Arabic', 'en': 'English'}
    language = lang_map.get(language, 'English')
    
    # Load document
    text = load_docx(doc_path)
    text_chunks = chunk_text(text)
    
    # Load Arabic embedding model
    embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Create FAISS index
    index, text_chunks = create_faiss_index(text_chunks, embedding_model)
    
    # Retrieve relevant context
    context = retrieve_top_k(user_query, index, text_chunks, embedding_model)
    
    # Load Llama 3-8B via Groq
    llama_model = init_chat_model("llama3-8b-8192", model_provider="groq")
    
    # Generate response
    response = generate_response(llama_model, user_query, context, language)
    
    return {"query": user_query, "response": response}

# Flask API endpoint
@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    doc_path = data.get("doc_path", "/content/YouLearnt_Ai_chat_robot[1].docx")  # Default document path
    user_query = data.get("question")
    
    if not user_query:
        return jsonify({"error": "No question provided"}), 400
    
    result = process_query(doc_path, user_query)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)