"""FastAPI wrapper - added after I got tired of running python REPL."""
from fastapi import FastAPI
from pydantic import BaseModel
from .retriever import get_qa_chain

app = FastAPI(title="rag-pipeline-eval")
ask_fn = None

class Query(BaseModel):
    question: str
    k: int = 4

@app.on_event("startup")
def load():
    global ask_fn
    # lazy load so uvicorn --reload doesn't re-index every time
    print("loading QA chain...")
    ask_fn = get_qa_chain()

@app.post("/ask")
def ask(q: Query):
    # TODO: add auth, rate limiting - this is wide open right now
    result = ask_fn(q.question)
    return result

@app.get("/health")
def health():
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
