import streamlit as st
import fitz
import pytesseract
from PIL import Image
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from sentence_transformers import SentenceTransformer, util


# ----------------------------------
# Tesseract path
# ----------------------------------
import os

if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ----------------------------------
# Clean text
# ----------------------------------
def clean_text(text: str):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------------
# PDF Loader with OCR
# ----------------------------------
def load_pdf_with_ocr(pdf_file):

    pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")
    docs = []

    for i, page in enumerate(pdf):

        native_text = page.get_text().strip()
        ocr_text = ""

        if len(native_text) < 50:

            pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
            img = Image.frombytes("RGB",[pix.width,pix.height],pix.samples)

            ocr_text = pytesseract.image_to_string(img)

        full_text = clean_text(native_text + " " + ocr_text)

        if full_text:

            docs.append(
                Document(
                    page_content=full_text,
                    metadata={"page": i+1}
                )
            )

    return docs


# ----------------------------------
# Sentence extractor
# ----------------------------------
def extract_best_snippet(query, retrieved_docs):

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    candidate_sentences = []

    for doc in retrieved_docs:

        sentences = re.split(r'(?<=[.!?])\s+', doc.page_content)

        for sent in sentences:

            sent = sent.strip()

            if len(sent) > 20:
                candidate_sentences.append((sent, doc.metadata))

    if not candidate_sentences:
        return "No answer found"

    query_embedding = model.encode(query, convert_to_tensor=True)

    sentence_texts = [s[0] for s in candidate_sentences]

    sentence_embeddings = model.encode(sentence_texts, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, sentence_embeddings)[0]

    best_idx = int(scores.argmax())

    best_sentence, meta = candidate_sentences[best_idx]

    return best_sentence, meta.get("page", "unknown")


# ----------------------------------
# Streamlit UI
# ----------------------------------

st.set_page_config(page_title="PDF Chatbot", layout="wide")

st.title("📄 PDF RAG Chatbot")

st.write("Upload a PDF and ask questions.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    with st.spinner("Processing PDF..."):

        documents = load_pdf_with_ocr(uploaded_file)

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
            embedding=embedding_model,
            persist_directory="chroma_db"
        )

        retriever = db.as_retriever(search_kwargs={"k":3})

    st.success("PDF processed successfully!")

    # ----------------------------
    # Chat UI
    # ----------------------------

    query = st.text_input("Ask a question from the PDF")

    if query:

        retrieved_docs = retriever.invoke(query)

        answer, page = extract_best_snippet(query, retrieved_docs)

        st.markdown("### 💡 Answer")

        st.success(answer)

        st.write(f"📄 Page: {page}")

        with st.expander("Retrieved Chunks"):

            for i, doc in enumerate(retrieved_docs, 1):

                st.write(f"Chunk {i} | Page {doc.metadata.get('page')}")
                st.write(doc.page_content[:500])
                st.write("---")
