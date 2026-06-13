import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


INDEX_PATH = "faiss_index"


# =====================================================
# Used for fixed PDFs stored locally
#
# Example:
# data/
#   ├── islr.pdf
#
# Creates FAISS once and reuses it.
# =====================================================
def create_or_load_vectorstore(
    documents,
    embeddings
):

    if os.path.exists(INDEX_PATH):

        print("Loading existing FAISS index...")

        return FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    print("Creating new FAISS index...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    splits = splitter.split_documents(
        documents
    )

    print(
        f"Total chunks: {len(splits)}"
    )

    vectorstore = FAISS.from_documents(
        splits,
        embeddings
    )

    vectorstore.save_local(
        INDEX_PATH
    )

    print("FAISS index saved.")

    return vectorstore


# =====================================================
# Used for Streamlit PDF Upload
#
# User uploads PDF
# ↓
# Create fresh FAISS
# ↓
# No saving/loading
#
# Prevents mixing data from
# multiple uploaded PDFs.
# =====================================================
def create_vectorstore(
    documents,
    embeddings
):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    splits = splitter.split_documents(
        documents
    )

    print(
        f"Total chunks: {len(splits)}"
    )

    vectorstore = FAISS.from_documents(
        splits,
        embeddings
    )

    return vectorstore