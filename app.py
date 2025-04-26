from flask import Flask, render_template, request, jsonify
import os
import json
import faiss
import langdetect
from langchain.chat_models import init_chat_model
from sentence_transformers import SentenceTransformer
from docx import Document
import numpy as np
from flask_cors import CORS
from dotenv import dotenv_values, load_dotenv

load_dotenv()
app = Flask(__name__, static_folder='static', template_folder='static')
CORS(app)

# Set Groq API Key directly (for testing)
#os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

def load_docx(filepath):
    """Extracts text from a Word document."""
    doc = Document(filepath)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def chunk_text(text, chunk_size=512):
    """Splits text into smaller chunks."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def create_faiss_index(text_chunks, embedding_model):
    """Creates a FAISS index from text chunks."""
    embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, text_chunks

def retrieve_top_k(query, index, text_chunks, embedding_model, k=3):
    """Retrieves top-k relevant chunks using FAISS, handles cases where no match is found."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    retrieved_texts = [text_chunks[i] for i in indices[0] if i < len(text_chunks)]
    
    if not retrieved_texts:
        return "لم أتمكن من العثور على إجابة لسؤالك، يمكنك التواصل مع فريق الدعم عبر support@youlearnt.com ."
    
    return " ".join(retrieved_texts)

def generate_response(model, query, context, history, language):
    """Generates a response using Llama 3-8B via Groq in the appropriate language."""
    # Format conversation history
    history_text = ""
    for turn in history:
        history_text += f"User: {turn['query']}\nBot: {turn['response']}\n"
    
    prompt = f"""
    Conversation History:
    {history_text}
    
    Context: {context}
    Question: {query}
    Answer (respond in {language}):
    """
    
    response = model.invoke(prompt)
    
    if not response:
        return "لم أتمكن من العثور على إجابة لسؤالك." if language == 'ar' else "I couldn't find an answer to your question."
    
    return response.content

def process_query(doc_path, user_query, history):
    """Main function to process a user query."""
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
    response = generate_response(llama_model, user_query, context, history, language)
    
    return {"query": user_query, "response": response}

@app.route('/')
def home():
    return render_template('front.html')

@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    user_query = data.get("question")
    history = data.get("history", [])  # Get conversation history
    doc_path = data.get("doc_path", r"C:\Users\George\Downloads\ai based\YouLearnt Ai chat robot.docx")
    
    if not user_query:
        return jsonify({"error": "No question provided"}), 400
    
    result = process_query(doc_path, user_query, history)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)