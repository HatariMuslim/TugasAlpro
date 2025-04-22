from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import re
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()

# Initialize Flask application
app = Flask(__name__)
app.secret_key = "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"
app.config["SESSION_PERMANENT"] = True

# Initialize model with adjusted parameters
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro", 
    temperature=0.8,
    max_tokens=250
)

# Updated professional system prompt
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "Anda adalah EduMate, asisten pembelajaran yang profesional dan bersahabat. "
            "Panduan interaksi:\n\n"
            "1. Gunakan bahasa yang natural dan sopan\n"
            "2. Tunjukkan pemahaman dan empati terhadap pertanyaan pengguna\n"
            "3. Berikan jawaban yang informatif namun mudah dipahami\n"
            "4. Akhiri dengan kata-kata motivasi yang natural\n\n"
            "Format jawaban:\n"
            "- Mulai dengan sapaan ramah\n"
            "- Tunjukkan pemahaman atas pertanyaan\n"
            "- Berikan penjelasan yang terstruktur\n"
            "- Sertakan saran praktis\n"
            "- Tidak semua jawaban perlu saran praktis\n"
            "- Tutup dengan dorongan positif\n\n"
            "- Berikan jawabnnya secara lengkap\n\n"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create LLMChain with memory
conversation_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

def get_current_time():
    return datetime.now().strftime("%H:%M")

def format_response_as_list(text):
    if not text:
        return "<p>Mohon maaf, saya perlu waktu sejenak. Silakan coba ajukan pertanyaan dengan cara yang berbeda.</p>"
    
    # Clean up any existing HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Format numbered points with clean bullets
    text = re.sub(r'(\d+)\.\s', r'<br>• ', text)
    text = re.sub(r'^<br>', '', text)
    
    # Add proper spacing and formatting
    text = re.sub(r'\.(?=\w)', '. ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<br>\s<br>', '<br>', text)
    
    # Add professional styling
    text = f'<p class="professional-message">{text}</p>'
    
    return text

def get_answer(question):
    try:
        result = conversation_chain.run(question=question)
        if result and result.strip():
            formatted_result = format_response_as_list(result.strip())
            return formatted_result
        else:
            logging.error("Received empty result from conversation_chain.")
            return "<p>Mohon maaf, saya membutuhkan waktu untuk memproses pertanyaan Anda. Silakan coba dengan pertanyaan yang lebih spesifik.</p>"
    except Exception as e:
        error_message = f"Error: {str(e)}"
        logging.error(error_message)
        return "<p>Mohon maaf, terjadi kendala teknis. Silakan coba beberapa saat lagi.</p>"

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/chat.html", methods=["GET", "POST"])
def chat():
    session.permanent = True
    if "history" not in session:
        session["history"] = []
    
    if request.method == "POST":
        try:
            data = request.get_json()
            question = data.get("message", "").strip()
            
            if question:
                logging.debug(f"Received question: {question}")
                answer = get_answer(question)
                logging.debug(f"Generated answer: {answer}")
                
                if answer:
                    current_time = get_current_time()
                    session["history"].append({
                        "role": "user",
                        "text": question,
                        "time": current_time
                    })
                    session["history"].append({
                        "role": "bot",
                        "text": answer,
                        "time": current_time
                    })
                    session.modified = True
                    return jsonify(answer=answer)
            
            return jsonify(
                answer="Mohon maaf, saya belum dapat memahami pertanyaan Anda. Silakan coba dengan pertanyaan yang lebih spesifik."
            )
            
        except Exception as e:
            logging.error(f"Error in chat route: {str(e)}")
            return jsonify(
                answer="Mohon maaf, terjadi kendala teknis. Silakan coba beberapa saat lagi."
            )
    
    return render_template("chat.html", history=session["history"])

@app.route("/clear_history", methods=["POST"])
def clear_history():
    session.pop("history", None)
    memory.clear()
    return redirect(url_for("chat"))

@app.before_request
def check_history_length():
    if len(session.get("history", [])) > 1000:
        session.pop("history", None)
        memory.clear()

if __name__ == "__main__":
    app.run(debug=True)