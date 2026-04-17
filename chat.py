import streamlit as st
import fitz
import re

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI


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
# Load LLM (Groq)
# -----------------------------
def load_llm():

    llm = ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=st.secrets["GROQ_API_KEY"],
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    return llm


# -----------------------------
# Generate Answer
# -----------------------------
def generate_answer(query, docs):

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the information from the context.

Context:
{context}

Question:
{query}

Answer:
"""

    llm = load_llm()

    response = llm.invoke(prompt)

    return response.content


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📄 Raghul RAG Chatbot")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")


# Detect new file upload
if uploaded_file:

    file_id = uploaded_file.name + str(uploaded_file.size)

    if st.session_state.get("file_id") != file_id:

        st.session_state.clear()

        st.session_state.file_id = file_id

        docs = load_pdf(uploaded_file)

        db = build_vector_db(docs)

        st.session_state.retriever = db.as_retriever(
            search_kwargs={"k": 3}
        )

        st.success("PDF processed successfully!")


# Ask question
if "retriever" in st.session_state:

    query = st.text_input("Ask a question")

    if query:

        retrieved_docs = st.session_state.retriever.invoke(query)

        answer = generate_answer(query, retrieved_docs)

        st.markdown("### 💡 Answer")

        st.success(answer)
