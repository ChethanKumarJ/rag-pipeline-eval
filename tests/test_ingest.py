"""basic tests - mostly checking chunking doesn't explode"""
from pathlib import Path
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter

def test_splitter():
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    long_text = "hello " * 1000
    chunks = splitter.split_text(long_text)
    assert len(chunks) > 1
    # overlap means chunks share some text - rough check
    assert len(chunks[0]) <= 600

def test_chunk_size_config():
    # regression test: I once set this to 5120 by typo and wondered why eval tanked
    from pathlib import Path
    import sys
    sys.path.insert(0, "src")
    # just check the constant exists
    import ingest
    assert ingest.CHUNK_SIZE == 512
