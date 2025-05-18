from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import sqlite3
from datetime import datetime
import fitz  # PyMuPDF
import os
import io
import nltk
from nltk.tokenize import word_tokenize

# Đường dẫn tới các thư mục chứa model đã fine-tune
vietnamese_model_dir = r"E:\AI\QABert\bert-finetuned-qa-vn"
english_model_dir = r"E:\AI\QABert\bert-finetuned-qa" 

# Tải NLTK data cần thiết
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Tải các language detection resources từ NLTK
nltk.download('stopwords')
from nltk.corpus import stopwords

# Cache cho models
model_cache = {}

def load_qa_model(model_dir):
    """Load model và tokenizer từ cache hoặc từ disk"""
    if model_dir not in model_cache:
        model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
        model_cache[model_dir] = qa_pipeline
    
    return model_cache[model_dir]

def detect_language(text):
    """Phát hiện ngôn ngữ của văn bản sử dụng NLTK"""
    # Danh sách từ dừng tiếng Việt
    vietnamese_stopwords = [
        "và", "là", "của", "có", "được", "trong", "đã", "với", "không", 
        "những", "một", "này", "đó", "các", "để", "cho", "người", "như"
    ]
    
    # Lấy stopwords tiếng Anh từ NLTK
    english_stopwords = set(stopwords.words('english'))
    
    # Tokenize text
    words = word_tokenize(text.lower())
    
    # Đếm số từ khóa tiếng Việt và tiếng Anh
    vi_count = sum(1 for word in words if word in vietnamese_stopwords)
    en_count = sum(1 for word in words if word in english_stopwords)
    
    # Xác định ngôn ngữ dựa trên tỉ lệ từ khóa
    if vi_count > en_count:
        return "vi"
    elif en_count > 0:
        return "en"
    else:
        # Phương pháp dự phòng: kiểm tra các ký tự đặc trưng tiếng Việt
        vietnamese_chars = set('àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ')
        if any(char in vietnamese_chars for char in text.lower()):
            return "vi"
        elif text.isascii():
            return "en"
        else:
            return "unknown"

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    text = ""
    try:
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

# Khởi tạo Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'

# Đảm bảo thư mục upload tồn tại
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Kết nối và tạo bảng SQLite
def get_db_connection():
    conn = sqlite3.connect("qa_history.db")
    conn.row_factory = sqlite3.Row
    return conn

# Khởi tạo database nếu chưa tồn tại
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Kiểm tra nếu bảng đã tồn tại
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='history'")
    table_exists = cursor.fetchone()

    if not table_exists:
        # Tạo bảng mới nếu chưa tồn tại
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                language TEXT,
                question TEXT,
                context TEXT,
                answer TEXT,
                score REAL
            )
        ''')
        conn.commit()
    else:
        # Kiểm tra xem cột 'language' đã tồn tại chưa
        cursor.execute("PRAGMA table_info(history)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Thêm cột 'language' nếu chưa tồn tại
        if 'language' not in columns:
            cursor.execute("ALTER TABLE history ADD COLUMN language TEXT")
            conn.commit()
    
    conn.close()

# Khởi tạo database khi app chạy
init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def answer_question():
    data = {}
    
    # Lấy nội dung từ form
    if request.form.get('context'):
        context = request.form.get('context')
    elif 'pdf_file' in request.files and request.files['pdf_file'].filename != '':
        pdf_file = request.files['pdf_file']
        context = extract_text_from_pdf(pdf_file)
    else:
        return jsonify({'error': 'No context or PDF provided'}), 400
    
    question = request.form.get('question', '')
    
    if not question or not context:
        return jsonify({'error': 'Missing question or context'}), 400
    
    # Tự động phát hiện ngôn ngữ
    language = detect_language(context)
    
    # Chọn model dựa trên ngôn ngữ phát hiện được
    if language == "vi":
        question_answerer = load_qa_model(vietnamese_model_dir)
        lang_name = "Tiếng Việt"
    elif language == "en":
        question_answerer = load_qa_model(english_model_dir)
        lang_name = "English"
    else:
        return jsonify({
            'error': 'Unsupported language detected. Only Vietnamese and English are supported.',
            'detected_language': 'Unknown'
        }), 400
    
    # Thực hiện trả lời câu hỏi
    result = question_answerer(question=question, context=context)
    answer = result['answer']
    score = float(result['score'])
    
    # Lưu vào SQLite
    conn = get_db_connection()
    conn.execute('''
        INSERT INTO history (timestamp, language, question, context, answer, score)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), lang_name, question, context, answer, score))
    conn.commit()
    conn.close()
    
    # Trả về kết quả dưới dạng JSON
    return jsonify({
        'answer': answer,
        'score': score,
        'detected_language': lang_name
    })

@app.route('/history')
def history():
    conn = get_db_connection()
    rows = conn.execute('SELECT timestamp, language, question, answer, score FROM history ORDER BY id DESC LIMIT 10').fetchall()
    conn.close()
    
    return render_template('history.html', history=rows)

if __name__ == '__main__':
    app.run(debug=True)
