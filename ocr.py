import os
import pytesseract
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import pipeline
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

# Configurar pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Configuración
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'documents.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'

# Asegúrate de que la carpeta de uploads exista
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)

# Modelo de la base de datos
class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(255), nullable=False)
    text_content = db.Column(db.Text, nullable=True)
    summary = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f'<Document {self.filename}>'

# Crear las tablas en la base de datos
with app.app_context():
    db.create_all()

# Inicializar modelos de NLP
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def perform_ocr(filepath):
    """Realiza OCR en el documento."""
    return pytesseract.image_to_string(Image.open(filepath))

def analyze_text(text):
    """Analiza el texto y genera un resumen."""
    return summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']

def answer_question(context, question):
    """Responde una pregunta basada en el contexto."""
    return qa_model(question=question, context=context)['answer']

# Rutas de la API
@app.route('/upload', methods=['POST'])
def upload_document():
    if 'document' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['document']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Realizar OCR y análisis
        text_content = perform_ocr(filepath)
        summary = analyze_text(text_content)
        
        new_document = Document(filename=filename, filepath=filepath, text_content=text_content, summary=summary)
        db.session.add(new_document)
        db.session.commit()
        
        return jsonify({'message': 'Documento subido y analizado con éxito', 'document_id': new_document.id}), 201

@app.route('/analyze/<int:doc_id>', methods=['GET'])
def analyze_document(doc_id):
    document = Document.query.get_or_404(doc_id)
    return jsonify({
        'message': f'Análisis del documento {doc_id}',
        'summary': document.summary,
        'full_text': document.text_content[:500] + '...'  # Primeros 500 caracteres
    })

@app.route('/question/<int:doc_id>', methods=['POST'])
def question_document(doc_id):
    document = Document.query.get_or_404(doc_id)
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'No se proporcionó ninguna pregunta'}), 400
    
    answer = answer_question(document.text_content, question)
    return jsonify({'message': f'Respuesta a la pregunta sobre el documento {doc_id}', 'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
