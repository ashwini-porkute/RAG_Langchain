import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community import embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import textwrap
import gradio as gr
import time

### 1. Data Ingestion source PDF and converting to documents and splitting to chunks:

loader = PyPDFLoader("attention.pdf")
pages = loader.load()
# print(pages[0])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(pages)
# print(len(chunks[0].page_content))

"""
Converting chunks to embeddings using embedding model foe ex: ollama, openAI, huggingface.
Here we are using ollama embeddings as they are open source and have good embedding size.
"""

vectorstore = Chroma.from_documents(
    documents=chunks,
    collection_name="ollama_embeds",
    embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
)

retriever = vectorstore.as_retriever()

GROQ_KEY = os.environ.get("GROQ_KEY")     

llm = ChatGroq(
            groq_api_key=GROQ_KEY,
            model_name='mixtral-8x7b-32768'
    )

rag_template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# response = rag_chain.invoke("What is attention?")
# print(textwrap.fill(response, width=80))


# Make the questions dynamic using a chat interface. Let's use gradio for this.
def process_question(user_question):
    start_time = time.time()

    # Directly using the user's question as input for rag_chain.invoke
    response = rag_chain.invoke(user_question)

    # Measure the response time
    end_time = time.time()
    response_time = f"Response time: {end_time - start_time:.2f} seconds."

    # Combine the response and the response time into a single string
    full_response = f"{response}\n\n{response_time}"

    return full_response

# Setup the Gradio interface
iface = gr.Interface(fn=process_question,
                     inputs=gr.Textbox(lines=2, placeholder="Type your question here..."),
                     outputs=gr.Textbox(),
                     title="GROQ CHAT",
                     description="Ask any question about your document, and get an answer along with the response time.")

# Launch the interface
iface.launch()
print("Done!!!")