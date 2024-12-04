import streamlit as st
import os
import torch
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import faiss
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModel

import time
from dotenv import load_dotenv
load_dotenv()
    
###load the Groq API key
groq_api_key=os.environ['GROQ_API_KEY']
os.environ["USER_AGENT"] = "YourAppName/1.0"

### Data Ingestion: (Collecting the source data and extracting the same to get converted to embeddings and then storing in vector database)
def extract_pdf_text(file_path):
    pdf_file = PdfReader(file_path)
    text_data = ''
    for pg in pdf_file.pages:
        text_data += pg.extractText()
    return text_data

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def compute_embeddings(texts):
    """Compute embeddings for a list of texts."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings.cpu().numpy()

# FAISS index initialization
embedding_dim = 384  # Dimension of embeddings (for all-MiniLM-L6-v2)
index = faiss.IndexFlatL2(embedding_dim)  # L2 distance for similarity

# Document metadata map
doc_map = {}

def add_pdf_to_faiss(text, pdf_path):
    """Process a PDF and add its content to FAISS."""
    # text = extract_text_from_pdf(pdf_path)
    doc_id = len(doc_map)
    doc_map[doc_id] = {"text": text, "file": os.path.basename(pdf_path)}
    embedding = compute_embeddings([text])
    index.add(embedding)

pdf_path = 'attention.pdf'
pdf_text = extract_pdf_text(pdf_path)
add_pdf_to_faiss(pdf_text, pdf_path)
print("PDFs are added to FAISS")

def search_faiss(query, k=1):
    """Retrieve top-k results from FAISS for a given query."""
    query_embedding = compute_embeddings([query])
    distances, indices = index.search(query_embedding, k)
    results = [{"file": doc_map[idx]["file"], "text": doc_map[idx]["text"], "distance": dist} 
               for idx, dist in zip(indices[0], distances[0])]
    return results

# Example query
query = "What is Attention?"
results = search_faiss(query)

print("\nQuery Results:")
for res in results:
    print(f"File: {res['file']}, Distance: {res['distance']}")
    print(f"Content Snippet: {res['text'][:200]}...\n")

