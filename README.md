# RAG_Langchain:

Use LLM_requirements.txt file to get all necessary libraries to run the codes.

Also, set environment variables in ~/.bashrc(Linux) file or .env(for windows) like
export GROQ_KEY=gsk_......................
export USER_AGENT="YourAppName/1.0"
export OPENAI_API_KEY=sk-proj-..........................

**Note:** create above keys from respective websites, and for ollama embedding model sign in on ollama.com and install ollama and download the model using ollama command as mentioned on website.

**###1. RAG1.py:**
=====================
- Langchain as orchestration framework.
- Input source is PDF.
- embeddings done by embedding model from ollama's "nomic-embed-text" model.
- VectorStore is Chromadb.
- LLM taken from GROQ.
- Gradio tool for web interface.


**###2. RAG2.py:**
=====================
- Langchain as orchestration framework.
- Input source is from web so, web scraping is done with beautifulsoup.
- embeddings done by embedding model from ollama's "nomic-embed-text" model.
- VectorStore is FAISS.
- LLM taken from GROQ.
- Ragas is used to evaluate the RAG on basis of groundtruths.


RAG is used to retrieve the updated information from external source and then augment these retrieved updated information and provides to user.

### RAG has below elements in workflow:
===========================================
- Data Ingestion (external data source, PDF, csv, Web etc...)
- Data Transformation (Converting to document chunks)
- Create Embeddings. (converting to vectors using embedding model)
- Store Embeddings in Vector Store (storing in vectorstore like Chromadb, FAISS, Weaviate etc...)
- Retreival based on User's query.(any retireval method like semantic search, match threshold etc...)
- Augment (augmentation done on basis of retrieved data and user query)
- Output to User based on user's query.(showcasing results to user via web interface like stramlit or Gradio or commandline chat interface.)
