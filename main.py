import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

# Load PDF
loader = PyPDFLoader("Gyton Medical Physiology.pdf")
docs = loader.load()

# Filter empty pages
filtered_docs = [
    doc for doc in docs
    if doc.page_content.strip() != ""
]

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

documents = text_splitter.split_documents(filtered_docs)

from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")


pc = Pinecone(api_key=api_key)   # ✅ This creates Pinecone client

index_name = "medical-chatbot-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

print("Index created successfully ✅")

# Vector Store
docsearch = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embedding,
    index_name=index_name
)

# Retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# LLM
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)


# 👇 Function Flask will call
def ask_question(question):

    response = qa_chain.invoke(question)

    return response["result"]