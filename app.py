import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import sqlite3
from datetime import datetime

# Đường dẫn tới các thư mục chứa model đã fine-tune
vietnamese_model_dir = r"E:\AI\QABert\bert-finetuned-qa-vn"
english_model_dir = r"E:\AI\QABert\bert-finetuned-qa"

# Tạo cache cho models và tokenizers
@st.cache_resource
def load_qa_model(model_dir):
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_pipeline

# Kết nối và tạo bảng SQLite
conn = sqlite3.connect("qa_history.db", check_same_thread=False)
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

# UI Streamlit
st.title("🤖 Question Answering System ")

# Chọn ngôn ngữ
language = st.radio("Chọn ngôn ngữ / Select language:", ["Tiếng Việt", "English"], index=0)

# Tải model tương ứng với ngôn ngữ được chọn
if language == "Tiếng Việt":
    question_answerer = load_qa_model(vietnamese_model_dir)
    context_placeholder = "Nhập đoạn văn bản (ngữ cảnh):"
    question_placeholder = "Nhập câu hỏi:"
    button_text = "Trả lời"
    thinking_text = "Đang suy nghĩ..."
    warning_text = "Vui lòng nhập cả đoạn văn bản và câu hỏi!"
else:
    question_answerer = load_qa_model(english_model_dir)
    context_placeholder = "Enter paragraph (context):"
    question_placeholder = "Enter question:"
    button_text = "Answer"
    thinking_text = "Thinking..."
    warning_text = "Please enter both the paragraph and the question!"

context = st.text_area(f"📚 {context_placeholder}", height=200)
question = st.text_input(f"❓ {question_placeholder}")

if st.button(button_text):
    if context.strip() == "" or question.strip() == "":
        st.warning(warning_text)
    else:
        with st.spinner(thinking_text):
            result = question_answerer(question=question, context=context)
            answer = result['answer']
            score = result['score']
            st.success(f"✅ {'Câu trả lời' if language == 'Tiếng Việt' else 'Answer'}: {answer}")
            st.info(f"📊 {'Độ tin cậy' if language == 'Tiếng Việt' else 'Confidence'}: {score:.4f}")
            
            # Lưu vào SQLite
            cursor.execute('''
                INSERT INTO history (timestamp, language, question, context, answer, score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), language, question, context, answer, score))
            conn.commit()

# Hiển thị lịch sử
st.markdown("---")
st.subheader("📜 " + ("Lịch sử câu hỏi" if language == "Tiếng Việt" else "Question History"))
cursor.execute("SELECT timestamp, language, question, answer, score FROM history ORDER BY id DESC LIMIT 10")
rows = cursor.fetchall()

if rows:
    for row in rows:
        timestamp, lang, q, a, s = row
        st.markdown(f"**🕒 {timestamp}**")
        st.markdown(f"**🌐 {lang}**")
        st.markdown(f"**❓ {'Câu hỏi' if lang == 'Tiếng Việt' else 'Question'}:** {q}")
        st.markdown(f"**✅ {'Câu trả lời' if lang == 'Tiếng Việt' else 'Answer'}:** {a}")
        st.markdown(f"**📊 {'Độ tin cậy' if lang == 'Tiếng Việt' else 'Confidence'}:** {s:.4f}")
        st.markdown("---")
else:
    st.info("Chưa có lịch sử" if language == "Tiếng Việt" else "No history found.")
