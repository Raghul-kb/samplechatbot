import streamlit as st
import fitz
import pytesseract
from PIL import Image
import re
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from sentence_transformers import SentenceTransformer, util


# -----------------------------
# Tesseract Path (Only Windows)
# -----------------------------
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# -----------------------------
# Clean text
# -----------------------------
def clean_text(text: str):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# Load PDF + OCR
# -----------------------------
def load_pdf_with_ocr(pdf_file):

    pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")
    docs = []

    for i, page in enumerate(pdf):

        native_text = page.get_text().strip()
        ocr_text = ""

        if len(native_text) < 50:

            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            try:
                ocr_text = pytesseract.image_to_string(img)
            except:
                ocr_text = ""

        full_text = clean_text(native_text + " " + ocr_text)

        if full_text:
            docs.append(
                Document(
                    page_content=full_text,
                    metadata={"page": i + 1}
                )
            )

    return docs


# -----------------------------
# Build Vector DB (cached)
# -----------------------------
@st.cache_resource
def build_vector_db(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    chunks = text_splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return db


# -----------------------------
# Sentence extractor
# -----------------------------
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def extract_best_snippet(query, retrieved_docs):

    model = load_sentence_model()

    candidate_sentences = []

    for doc in retrieved_docs:

        sentences = re.split(r'(?<=[.!?])\s+', doc.page_content)

        for sent in sentences:

            sent = sent.strip()

            if len(sent) > 20:
                candidate_sentences.append((sent, doc.metadata))

    if not candidate_sentences:
        return "No answer found", "unknown"

    query_embedding = model.encode(query, convert_to_tensor=True)

    sentence_texts = [s[0] for s in candidate_sentences]

    sentence_embeddings = model.encode(sentence_texts, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, sentence_embeddings)[0]

    best_idx = int(scores.argmax())

    best_sentence, meta = candidate_sentences[best_idx]

    return best_sentence, meta.get("page", "unknown")


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")

st.title("📄 PDF RAG Chatbot")
st.write("Upload a PDF and ask questions.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")


if uploaded_file:

    with st.spinner("Processing PDF..."):

        documents = load_pdf_with_ocr(uploaded_file)

        db = build_vector_db(documents)

        retriever = db.as_retriever(search_kwargs={"k": 3})

    st.success("PDF processed successfully!")

    # -----------------------------
    # Chat history
    # -----------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    query = st.chat_input("Ask something about the PDF")

    if query:

        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        retrieved_docs = retriever.invoke(query)

        answer, page = extract_best_snippet(query, retrieved_docs)

        response = f"{answer}\n\n📄 **Page:** {page}"

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        # Show retrieved chunks
        with st.expander("Retrieved Chunks"):

            for i, doc in enumerate(retrieved_docs, 1):

                st.write(f"Chunk {i} | Page {doc.metadata.get('page')}")
                st.write(doc.page_content[:500])
                st.write("---")
