import streamlit as st
import fitz
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from sentence_transformers import SentenceTransformer, util


# -----------------------------
# Clean text
# -----------------------------
def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# Load PDF
# -----------------------------
def load_pdf(file):

    pdf = fitz.open(stream=file.read(), filetype="pdf")
    docs = []

    for i, page in enumerate(pdf):

        text = page.get_text()
        text = clean_text(text)

        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"page": i+1}
                )
            )

    return docs


# -----------------------------
# Build Vector DB
# -----------------------------
def build_vector_db(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return db


# -----------------------------
# Sentence model
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# -----------------------------
# Extract answer
# -----------------------------
def extract_answer(query, docs):

    model = load_model()

    sentences = []

    for d in docs:

        parts = re.split(r'(?<=[.!?])\s+', d.page_content)

        for s in parts:
            if len(s) > 20:
                sentences.append(s)

    if not sentences:
        return "No answer found."

    q_emb = model.encode(query, convert_to_tensor=True)
    s_emb = model.encode(sentences, convert_to_tensor=True)

    scores = util.cos_sim(q_emb, s_emb)[0]

    best = sentences[int(scores.argmax())]

    return best


# -----------------------------
# UI
# -----------------------------
st.title("📄 PDF RAG Chatbot")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")


# Detect new file
if uploaded_file:

    file_id = uploaded_file.name + str(uploaded_file.size)

    if st.session_state.get("file_id") != file_id:

        # Reset old retriever
        st.session_state.clear()

        st.session_state.file_id = file_id

        docs = load_pdf(uploaded_file)

        db = build_vector_db(docs)

        st.session_state.retriever = db.as_retriever(search_kwargs={"k":3})

        st.success("PDF processed successfully!")


# Ask question
if "retriever" in st.session_state:

    query = st.text_input("Ask a question")

    if query:

        retrieved_docs = st.session_state.retriever.invoke(query)

        answer = extract_answer(query, retrieved_docs)

        st.markdown("### 💡 Answer")
        st.success(answer)
