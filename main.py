import os
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Gyton Medical Physiology.pdf")
docs = loader.load()
docs

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 200)
document = text_splitter.split_documents(docs)
document

from dotenv import load_dotenv
load_dotenv() ##loading all the env variables

groq_apik=os.getenv("GROQ_API_KEY")
groq_apik

##load a model

from langchain_groq import ChatGroq
model= ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_apik)
model

from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

from langchain_community.vectorstores import FAISS

vector = FAISS.from_documents(document, embedding)
vector

retriever = vector.as_retriever(
    search_kwargs={"k": 4}
)

query = "Explain cardiac output in simple terms"

retrieved_docs = retriever.invoke(query)

# Combine text
context = "\n\n".join([doc.page_content for doc in retrieved_docs])
context

from langchain_core.messages import HumanMessage

prompt = f"""
You are a medical assistant.
Answer the question strictly using the context below.

Context:
{context}

Question:
{query}

Answer:
"""

response = model.invoke([HumanMessage(content=prompt)])
print(response.content)

