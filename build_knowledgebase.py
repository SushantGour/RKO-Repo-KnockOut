import os

from dotenv import load_dotenv
load_dotenv()

from langchain_pinecone import PineconeVectorStore

from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

import os
import tempfile
import glob
from git import Repo
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
import tree_sitter_javascript as tsjavascript
import tree_sitter_cpp as tscpp
import tree_sitter_go as tsgo
import tree_sitter_rust as tsrust
from tree_sitter import Language, Parser
from langchain_core.documents import Document

# -----------------------
# Map extensions to Language objects
# -----------------------
LANGUAGE_MAP = {
    ".py": tspython.language(),
    ".java": tsjava.language(),
    ".js": tsjavascript.language(),
    ".cpp": tscpp.language(),
    ".c": tscpp.language(),
    ".go": tsgo.language(),
    ".rs": tsrust.language(),
}

# -----------------------
# Extract code snippets from the syntax tree
# -----------------------
def extract_code_snippets(code: str, file_ext: str):
    language = Language(LANGUAGE_MAP.get(file_ext))
    if not language:
        # Fallback: return whole code chunk as single snippet if unsupported language
        return [code]

    parser = Parser(language)
    tree = parser.parse(bytes(code, "utf8"))
    root = tree.root_node

    snippets = []

    def walk(node):
        # Different languages use different node types, but commonly:
        # functions, methods, classes
        if node.type in (
            "function_definition",    # Python
            "method_definition",      # Java, C++
            "function_declaration",   # JS, C++
            "class_declaration",      # Java, Python, C++, JS
            "method_declaration",     # Java
            "function",               # Go
            "struct_declaration",     # Rust, Go
            "enum_declaration",       # Rust
        ):
            start, end = node.start_byte, node.end_byte
            snippet = code[start:end]
            snippets.append(snippet)

        for child in node.children:
            walk(child)

    walk(root)

    # fallback if nothing extracted
    if not snippets:
        snippets.append(code)

    return snippets

def chunk_snippet(snippet: str, max_bytes: int = 2 * 1024 * 1024): # max_bytes = 2MB. Pinecone's document size limit is 4MB. So, we are reserving 2MB for out page content and the rest 2MB for metadata, pinecone vector, JSON encoding overheads, network formatting, etc.
    """Ensure the encoded snippet fits within the byte limit."""
    encoded = snippet.encode("utf-8")
    chunks = []
    for i in range(0, len(encoded), max_bytes):
        chunk_bytes = encoded[i:i + max_bytes]
        chunks.append(chunk_bytes.decode("utf-8", errors="ignore"))
    return chunks

MAX_DOC_SIZE_BYTES = 3900000  # a safe buffer under 4MB (Pinecone's document size limit is 4MB)

def is_too_large(doc):
    import json
    # Simulate Pinecone payload structure: vector + metadata
    dummy_vector = [0.0] * 1024  # typical vector size
    payload = {
        "id": "dummy_id",
        "values": dummy_vector,
        "metadata": doc.metadata | {"text": doc.page_content}
    }
    size = len(json.dumps(payload).encode("utf-8"))
    return size > MAX_DOC_SIZE_BYTES

# Batch upload documents in pinecone. Each batch has 100 documents. This is to make sure that Pinecone doesn't give an error saying too many documents uploaded in one request
def batch_upload(docs, embeddings, batch_size=100):
    batch_number=0
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]

        try:
            PineconeVectorStore.from_documents(
                documents=batch,
                embedding=embeddings,
                index_name=os.environ["INDEX_NAME"]
            )

            print(f"✅ uploaded batch {batch_number} to pinecone")
            batch_number += 1
        except Exception as e:
            print(f"❌ Error uploading batch {i // batch_size + 1}: {e}")
            batch_number += 1

def build_knowledgebase(url : str):
    print("Building knowledgebase...")

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

    # ✅ Instantiate vectorstore for similarity search
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=os.environ["INDEX_NAME"],
        embedding=embeddings
    )

    # ✅ Check if repo already exists in knowledgebase
    try:
        existing = vectorstore.similarity_search(
            query="dummy query",  # required but irrelevant since we filter by metadata
            k=1,
            filter={"repo_url":url}
        )
        if existing:
            print("🔁 Repo already exists in vector DB")
            return "already_exists"
    except Exception as e:
        print(f"[Warning] Skipping repository already exists similarity search check due to error: {e}")

    # add repository to knowledgebase

    # Clone repo to temp dir
    temp_dir = tempfile.mkdtemp()
    print(f"Cloning {url} into {temp_dir} ...")
    Repo.clone_from(url, temp_dir)

    print("Successfully cloned repository")

    # Gather all source files recursively
    all_files = glob.glob(f"{temp_dir}/**/*.*", recursive=True)

    print("Building documents from the code files in the repository...")

    documents = []

    for file_path in all_files:
        ext = os.path.splitext(file_path)[1]
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()

            snippets = extract_code_snippets(code, ext)

            for snippet in snippets:
                for chunk in chunk_snippet(snippet):
                    doc = Document(page_content=chunk,
                    metadata={"file_path": file_path, "repo_url": url})  # Add repo URL as metadata
                    if not is_too_large(doc):
                        documents.append(doc)
                    else:
                        print(f"Skipped chunk from {file_path} because it's too large")

        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")

    print(f"Successfully built {len(documents)} documents...")

    # Create Embeddings from the documents and store in Pinecone

    print(f"Ingesting {len(documents)} documents into Pinecone Vector Store")

    # Batch upload documents to Pinecone
    batch_upload(documents, embeddings)

# if __name__ == "__main__":
#     build_knowledgebase("https://github.com/SushantGour/RKO-Repo-KnockOut")