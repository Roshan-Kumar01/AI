import os
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
)
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ==========================
# Load Environment Variables
# ==========================
load_dotenv("../.env")

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY not found in .env")


# ==========================
# Load Documents
# ==========================
def load_document(docs_path="docs"):
    """Load all .txt documents from the docs directory."""

    print(f"Loading documents from '{docs_path}'...\n")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory '{docs_path}' does not exist.")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={
            "encoding": "utf-8",
            "autodetect_encoding": True,
        },
    )

    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(f"No .txt files found inside '{docs_path}'.")

    print(f"Loaded {len(documents)} document(s).\n")

    for i, doc in enumerate(documents[:2]):
        print(f"Document {i + 1}")
        print(f"Source : {doc.metadata['source']}")
        print(f"Length : {len(doc.page_content)} characters")
        print(f"Preview:\n{doc.page_content[:150]}")
        print("-" * 60)

    return documents


# ==========================
# Split Documents
# ==========================
def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    """Split documents into chunks."""

    print("\nSplitting documents...\n")

    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.\n")

    for i, chunk in enumerate(chunks[:5]):
        print(f"Chunk {i + 1}")
        print(f"Source : {chunk.metadata['source']}")
        print(f"Length : {len(chunk.page_content)}")
        print(chunk.page_content[:200])
        print("-" * 60)

    if len(chunks) > 5:
        print(f"...and {len(chunks) - 5} more chunks.\n")

    return chunks


# ==========================
# Create Vector Store
# ==========================
def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create Chroma vector store."""

    print("\nCreating embeddings...\n")

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata={
            "hnsw:space": "cosine"
        },
    )

    print(f"\nVector store saved to '{persist_directory}'")

    return vectorstore


# ==========================
# Load Existing Vector Store
# ==========================
def load_vector_store(persist_directory="db/chroma_db"):
    """Load an existing Chroma vector store."""

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_metadata={
            "hnsw:space": "cosine"
        },
    )

    return vectorstore


# ==========================
# Main
# ==========================
def main():

    print("=" * 60)
    print("RAG Document Ingestion Pipeline")
    print("=" * 60)

    docs_path = "docs"
    persist_directory = "db/chroma_db"

    # If database already exists
    if os.path.exists(persist_directory):

        print("\nExisting vector store found.\n")

        vectorstore = load_vector_store(persist_directory)

        print(
            f"Loaded vector store with "
            f"{vectorstore._collection.count()} embeddings."
        )

        return vectorstore

    print("\nNo vector store found.\n")

    documents = load_document(docs_path)

    chunks = split_documents(documents)

    vectorstore = create_vector_store(
        chunks,
        persist_directory,
    )

    print("\nIngestion completed successfully.")

    return vectorstore


if __name__ == "__main__":
    main()