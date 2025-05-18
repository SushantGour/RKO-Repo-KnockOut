import os

from dotenv import load_dotenv

load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer


def get_answer(query : str):
    print("Retrieving....")

    llm = ChatOllama(model='mistral', temperature=0, base_url=os.getenv("OLLAMA_API_URL"))

    # Load the embedding model
    model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')  # model with same no. of

    # dimensions as our pinecone index, i.e., 1024

    # Create custom embedding class
    class SentenceTransformersEmbeddings(Embeddings):
        def __init__(self, model):
            self.model = model

        def embed_documents(self, texts):
            return self.model.encode(texts, convert_to_tensor=True).tolist()

        def embed_query(self, text):
            return self.model.encode(text, convert_to_tensor=True).tolist()

    # Initialize the custom embeddings class
    embeddings = SentenceTransformersEmbeddings(model=model)

    vectorstore = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embeddings)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(),
                                             combine_docs_chain=combine_docs_chain)

    # This is equivalent to : (input + context from vectorstore) | retrieval prompt | llm

    print("Doing similarity search and getting the context and then feeding the question and context to the llm")

    result = retrieval_chain.invoke(input={"input": query})

    print(result)

    return result["answer"]