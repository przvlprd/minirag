#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import chromadb
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from constants import model_path, n_ctx, n_gpu_layers, n_batch


class RAGBot:

    def __init__(self):
        # Import LLM
        self.llm = self.load_model()

        # Initialize vectordb
        self.client = chromadb.PersistentClient(path="/docs/chromadb")
        # Load or create collection / files
        self.collection = self.client.get_or_create_collection(
            name="docs",
            # embedding_function=self.embedding_function()  # Test if it works
        )

        self.langchain_chroma = Chroma(
            client=self.client,
            collection_name="docs",
            embedding_function=self.embedding_function()
        )

    @staticmethod
    def load_model():
        # Callbacks support token-wise streaming
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        llm = LlamaCpp(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            callback_manager=callback_manager,  # For streaming output
            verbose=True,  # Verbose is required to pass to callback manager
        )

        return llm

    def add_texts_to_vectordb(self):
        # Each document must have a unique associated id
        docs = self.load_raw_text_chunks()
        # for doc in docs:
        #     self.collection.add(
        #         ids=[str(uuid.uuid1())], metadatas=doc.metadata,
        #         documents=doc.page_content
        #     )
        self.langchain_chroma.add_documents(docs)

    @staticmethod
    def embedding_function():
        # Define embedding model
        embeddings = LlamaCppEmbeddings(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch
        )
        return embeddings

    @staticmethod
    def load_raw_text_chunks(file_path="docs/raw.txt"):
        # Load Doc
        # ToDo: so far only a raw text document is used
        loader = TextLoader(file_path,  encoding='utf-8')
        docs = loader.load()

        # Transform into chunks
        text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=20)
        texts = text_splitter.split_documents(docs)

        # Generate IDs for each text
        # text_ids = list(range(len(texts)))

        return texts  # text_ids

    @staticmethod
    def get_prompt_from_prompt_template():
        # Craft a prompt template based on the raw data we loaded
        template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try 
        to make up an answer.
        
        {context}
        Question: {question}
        Answer:"""

        # Context will be the similar doc and question will be our query
        prompt = PromptTemplate.from_template(template)

        return prompt

    def main(self):
        while True:
            # Set query according to your raw data
            query = input("Ask a question about your docs:\n")

            # Get the doc
            similar_doc = self.langchain_chroma.similarity_search(query, k=1)

            # Set the context
            context = similar_doc[0].page_content

            # Use LLM to generate answer from context
            query_llm = LLMChain(
                llm=self.llm,
                prompt=self.get_prompt_from_prompt_template()
            )

            print(query_llm.run({"context": context, "question": query}))


if __name__ == '__main__':
    bot = RAGBot()
    text_chunks = bot.load_raw_text_chunks("docs/raw.txt")
    # FixMe: Only run one time
    bot.add_texts_to_vectordb()

    # Loop, exit with Ctrl+C
    bot.main()
