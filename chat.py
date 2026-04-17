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
def load_pdf(pdf_file):

    pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")

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
def build_vector_db(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    chunks = splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return db


# -----------------------------
# Sentence model
# -----------------------------
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# -----------------------------
# Extract answer
# -----------------------------
def extract_best_snippet(query, retrieved_docs):

    model = load_sentence_model()

    sentences = []
    metas = []

    for doc in retrieved_docs:

        s = re.split(r'(?<=[.!?])\s+', doc.page_content)

        for x in s:
            if len(x) > 20:
                sentences.append(x)
                metas.append(doc.metadata)

    if not sentences:
        return "No answer found", "unknown"

    q_emb = model.encode(query, convert_to_tensor=True)
    s_emb = model.encode(sentences, convert_to_tensor=True)

    scores = util.cos_sim(q_emb, s_emb)[0]

    idx = int(scores.argmax())

    return sentences[idx], metas[idx]["page"]


# -----------------------------
# UI
# -----------------------------
st.title("📄 PDF RAG Chatbot")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")


# When new file uploaded
if uploaded_file:

    # Always rebuild DB for uploaded file
    documents = load_pdf(uploaded_file)

    db = build_vector_db(documents)

    st.session_state.retriever = db.as_retriever(search_kwargs={"k":3})

    st.success("PDF processed successfully!")


# Ask question
if "retriever" in st.session_state:

    query = st.text_input("Ask a question from the PDF")

    if query:

        retrieved_docs = st.session_state.retriever.invoke(query)

        answer, page = extract_best_snippet(query, retrieved_docs)

        st.markdown("### 💡 Answer")

        st.success(answer)

        st.write(f"📄 Page: {page}")
