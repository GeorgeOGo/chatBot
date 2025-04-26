from flask import Flask, render_template, request, jsonify
import os
import json
import langdetect
from langchain.chat_models import init_chat_model
from docx import Document
from flask_cors import CORS
from dotenv import dotenv_values, load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
app = Flask(__name__, static_folder='static', template_folder='static')
CORS(app)

def load_docx(filepath):
    """Extracts text from a Word document with a size limit."""
    try:
        doc = Document(filepath)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        # Limit text size to reduce memory usage (e.g., first 5,000 characters)
        if len(text) > 5000:
            logger.warning("Document too large, truncating to 5,000 characters.")
            text = text[:5000]
        return text
    except Exception as e:
        logger.error(f"Error loading document: {str(e)}")
        return f"Error loading document: {str(e)}"

def generate_response(model, query, context, history, language):
    """Generates a response using Llama 3-8B via Groq in the appropriate language."""
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
    language = langdetect.detect(user_query)
    lang_map = {'ar': 'Arabic', 'en': 'English'}
    language = lang_map.get(language, 'English')
    
    # Load document text directly as context
    context = load_docx(doc_path)
    
    # Load Llama 3-8B via Groq
    llama_model = init_chat_model("llama3-8b-8192", model_provider="groq")
    response = generate_response(llama_model, user_query, context, history, language)
    
    return {"query": user_query, "response": response}

@app.route('/')
def home():
    return render_template('front.html')

@app.route('/health')
def health():
    return 'OK', 200

@app.route('/api/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        user_query = data.get("question")
        history = data.get("history", [])
        doc_path = data.get("doc_path", r"YouLearnt Ai chat robot.docx")
        
        if not user_query:
            return jsonify({"error": "No question provided"}), 400
        
        result = process_query(doc_path, user_query, history)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in /api/ask: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)