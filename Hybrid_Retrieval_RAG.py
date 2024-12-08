import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import BM25Retriever, EnsembleRetriever


HF_TOKEN = os.environ.get("HF_TOKEN")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HF_TOKEN

dataset_folder_path = "/home/ashwini/TriedProjects/NLP/RAG/bbc-fulltext/bbc/entertainment" ### provide dataset path here

documents=[]
for file in os.listdir(dataset_folder_path):
  loader=TextLoader(os.path.join(dataset_folder_path, file))
  documents.extend(loader.load())
# print(documents[:3])
  
text_splitter=RecursiveCharacterTextSplitter(chunk_size=512,chunk_overlap=50)
text_splits=text_splitter.split_documents(documents)
print(len(text_splits))

embeddings=HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN,
    model_name='BAAI/bge-base-en-v1.5'
)

vectorstore = FAISS.from_documents(text_splits[:100], embeddings)

### using both BM25 and vectorembedding retriver to get accurate retreivals:
retriever_vectordb = vectorstore.as_retriever(search_kwargs={"k": 5})
keyword_retriever = BM25Retriever.from_documents(text_splits)
keyword_retriever.k =  5
ensemble_retriever = EnsembleRetriever(retrievers=[retriever_vectordb,keyword_retriever],
                                       weights=[0.5, 0.5])

query="How many cafes were closed in 2004?"
docs_rel=ensemble_retriever.get_relevant_documents(query)
# print(docs_rel)

### Augmentation:
from langchain.llms import HuggingFaceHub
model=HuggingFaceHub(repo_id='HuggingFaceH4/zephyr-7b-alpha',
                     model_kwargs={"temperature":0.5,"max_new_tokens":512,"max_length":64}
)

### Compressing hybrid retriever response:
compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=ensemble_retriever
)
compressed_docs = compression_retriever.get_relevant_documents(query)

template = """
<|system|>>
You are an AI Assistant that follows instructions extremely well.
Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in CONTEXT

CONTEXT: {context}
</s>
<|user|>
{query}
</s>
<|assistant|>
"""

prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

chain = (
    {"context": compression_retriever, "query": RunnablePassthrough()}
    | prompt
    | model
    | output_parser
)

# query="How many cafes were closed in 2004 in China?"
query="who was the star of the Oscar-nominated film The Chorus?"
response = chain.invoke(query)
print(response)