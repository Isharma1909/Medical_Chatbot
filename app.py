import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit UI
st.title("🩺 Medical Chatbot")
st.write("Ask questions from the medical textbook")

# Load & process PDF (cached)
@st.cache_resource
def prepare_vector():
    loader = PyPDFLoader("Gyton Medical Physiology.pdf")
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)

vector = prepare_vector()
retriever = vector.as_retriever(k=4)

# Load LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY
)

# User input
question = st.text_input(
    "Enter your question",
    "Explain cardiac output in simple terms"
)

if st.button("Ask"):
    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a medical assistant.
Use ONLY the information below to answer.

Context:
{context}

Question:
{question}

Answer:
"""

    answer = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    st.subheader("Answer")
    st.write(answer.content)

    with st.expander("Source Pages"):
        for doc in docs:
            st.write("Page:", doc.metadata.get("page", "N/A"))
