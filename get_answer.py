import os

from dotenv import load_dotenv

load_dotenv()
from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

from langchain_google_genai import ChatGoogleGenerativeAI


def get_answer(query : str):
    print("Retrieving....")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

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

    # Set a clear and courteous context-only instruction
    retrieval_qa_chat_prompt.messages[0].prompt.template = (
            "You are a helpful assistant. Please answer the question using only the information provided in the context below, "
            "which has been retrieved from a trusted vector database. Do not rely on your own pretraining knowledge, "
            "and avoid making assumptions or generating information that isn't supported by the context. "
            "If the context does not contain enough information to answer the question, respond with: "
            "'I'm sorry, but based on the provided context, I don't have enough information to answer that question.'\n\n"
            + retrieval_qa_chat_prompt.messages[0].prompt.template
    )

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(),
                                             combine_docs_chain=combine_docs_chain)

    # This is equivalent to : (input + context from vectorstore) | retrieval prompt | llm

    print("Doing similarity search and getting the context and then feeding the question and context to the llm")

    result = retrieval_chain.invoke(input={"input": query})

    print(result)

    return result["answer"]