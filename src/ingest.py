"""Ingest docs into Chroma - v1, pretty naive."""
import argparse
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 512 worked better than 1024 on my test set, don't ask me why
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

def ingest(docs_dir: str, db_dir: str):
    docs_path = Path(docs_dir)
    print(f"loading from {docs_path}...")

    # TODO: add PDF loader, for now just md/txt
    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        show_progress=True,
    )
    docs = loader.load()
    print(f"loaded {len(docs)} docs")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"split into {len(chunks)} chunks")

    # this takes a while on first run, embeddings aren't free lol
    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(),
        persist_directory=db_dir,
    )
    print(f"wrote to {db_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--db", default="./index")
    args = ap.parse_args()
    ingest(args.docs, args.db)
