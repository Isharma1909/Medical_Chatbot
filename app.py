import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()
groq_apik = os.getenv("GROQ_API_KEY")

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Medical RAG Chatbot", layout="wide")
st.title("🩺 Medical Chatbot (RAG)")
st.caption("PDF-based medical assistant using Groq + FAISS")

# ------------------------------
# Cache heavy operations
# ------------------------------
@st.cache_resource
def load_vectorstore():
    loader = PyPDFLoader("Gyton Medical Physiology.pdf")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    document = text_splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector = FAISS.from_documents(document, embedding)
    return vector

# ------------------------------
# Load Vector Store & Model
# ------------------------------
vector = load_vectorstore()

retriever = vector.as_retriever(search_kwargs={"k": 4})

model = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_apik
)

# ------------------------------
# User Input
# ------------------------------
query = st.text_input(
    "Ask a medical question:",
    placeholder="Explain cardiac output in simple terms"
)

if st.button("Ask") and query:
    with st.spinner("Retrieving answer..."):

        # Retrieve documents
        retrieved_docs = retriever.invoke(query)

        # Combine context
        context = "\n\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

        # Prompt (same as your code)
        prompt = f"""
You are a medical assistant.
Answer the question strictly using the context below.

Context:
{context}

Question:
{query}

Answer:
"""

        response = model.invoke(
            [HumanMessage(content=prompt)]
        )

        # Display answer
        st.subheader("🧠 Answer")
        st.write(response.content)

        # Show sources
        with st.expander("📄 Source Pages"):
            for doc in retrieved_docs:
                st.markdown(
                    f"- Page **{doc.metadata.get('page', 'N/A')}**"
                )
