import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Đường dẫn tới thư mục chứa model đã fine-tune
save_dir = "bert-finetuned-qa"

# Load model và tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(save_dir)
tokenizer = AutoTokenizer.from_pretrained(save_dir)

# Tạo pipeline
question_answerer = pipeline("question-answering", model=model, tokenizer=tokenizer)

# UI Streamlit
st.title("🤖 Question Answering System with BERT")

context = st.text_area("📚 Enter paragraph (context):", height=200)
question = st.text_input("❓ Enter question:")

if st.button("Answer"):
    if context.strip() == "" or question.strip() == "":
        st.warning("Please enter both the paragraph and the question.!")
    else:
        with st.spinner("Thinking..."):
            result = question_answerer(question=question, context=context)
            st.success(f"✅ Answer: {result['answer']}")
            st.info(f"📊 Confidence: {result['score']:.4f}")
