import streamlit as st
import fitz
import re
from PIL import Image
import pytesseract
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from sentence_transformers import SentenceTransformer, util


# Windows OCR only
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


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


@st.cache_resource
def build_db(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    chunks = splitter.split_documents(docs)

    embed = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(chunks, embed)

    return db


@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def answer_query(query, docs):

    model = load_model()

    sentences = []
    metas = []

    for d in docs:

        s = re.split(r'(?<=[.!?])\s+', d.page_content)

        for x in s:
            if len(x) > 20:
                sentences.append(x)
                metas.append(d.metadata)

    q_emb = model.encode(query, convert_to_tensor=True)
    s_emb = model.encode(sentences, convert_to_tensor=True)

    scores = util.cos_sim(q_emb, s_emb)[0]

    idx = int(scores.argmax())

    return sentences[idx], metas[idx]["page"]


# ---------------- UI ----------------

st.title("PDF Chatbot")

file = st.file_uploader("Upload PDF", type="pdf")

if file:

    docs = load_pdf(file)

    db = build_db(docs)

    retriever = db.as_retriever(search_kwargs={"k":3})

    q = st.text_input("Ask a question")

    if q:

        retrieved = retriever.invoke(q)

        ans, page = answer_query(q, retrieved)

        st.success(ans)

        st.write("Page:", page)
