"""
ingest.py — Load documents into ChromaDB vector store.

Supports: PDF, DOCX, TXT, CSV
Usage:
    python ingest.py --docs_dir ./documents
    python ingest.py --docs_dir ./documents --reset
"""

import os

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"



import argparse
import logging

from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "mxbai-embed-large")
CHROMA_DIR      = os.getenv("CHROMA_PERSIST_DIR", "./chromadb_data")
COLLECTION      = os.getenv("CHROMA_COLLECTION", "nspcl_docs")

LOADERS = {
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt":  TextLoader,
    ".csv":  CSVLoader,
}


def load_documents(docs_dir: str):
    docs = []
    path = Path(docs_dir)
    files = list(path.rglob("*"))
    logger.info(f"Found {len(files)} files in {docs_dir}")

    for f in files:
        loader_cls = LOADERS.get(f.suffix.lower())
        if loader_cls:
            try:
                loader = loader_cls(str(f))
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata["source"] = f.name
                docs.extend(loaded)
                logger.info(f"  Loaded: {f.name} ({len(loaded)} chunks)")
            except Exception as e:
                logger.warning(f"  Failed to load {f.name}: {e}")

    logger.info(f"Total documents loaded: {len(docs)}")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Total chunks after splitting: {len(chunks)}")
    return chunks


def ingest(docs_dir: str, reset: bool = False):
    embedding = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
        show_progress=True,
    )

    if reset:
        import shutil
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
            logger.info("Existing ChromaDB cleared.")

    docs   = load_documents(docs_dir)
    chunks = split_documents(docs)

    logger.info("Embedding and storing in ChromaDB...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION,
    )
    logger.info(f"Done. {vectordb._collection.count()} vectors stored in ChromaDB.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", default="./documents", help="Path to documents folder")
    parser.add_argument("--reset", action="store_true", help="Clear existing ChromaDB before ingesting")
    args = parser.parse_args()
    ingest(args.docs_dir, args.reset)
