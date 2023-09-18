#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from constants import model_path, n_ctx, n_gpu_layers, n_batch


# Define model
llm = LlamaCpp(
    model_path=model_path,
    n_ctx=n_ctx,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch
)

# Define embedding model
embeddings = LlamaCppEmbeddings(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch
)

# Load Doc
# ToDo: so far only a raw text document is used
loader = TextLoader("docs/raw.txt",  encoding='utf-8')
docs = loader.load()

# Transform into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20)
texts = text_splitter.split_documents(docs)

# Create a chroma vectorstore from a list of documents
db = Chroma.from_documents(texts, embeddings)

# Craft a prompt template based on the raw data we loaded
template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make 
up an answer.
{context}
Question: {question}
Answer:"""

# Context will be the similar doc and question will be our query
prompt = PromptTemplate.from_template(template)

# Set query according to your raw data
query = input("Ask a question about your docs:\n")

# Get the doc
similar_doc = db.similarity_search(query, k=1)

# Set the context
context = similar_doc[0].page_content

# Use LLM to generate answer from context
query_llm = LLMChain(llm=llm, prompt=prompt)
response = query_llm.run({"context": context, "question": query})

print(response)
