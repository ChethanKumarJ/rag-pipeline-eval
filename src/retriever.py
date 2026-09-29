"""Retriever + QA chain."""
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

SYSTEM = """You are a helpful assistant. Answer using ONLY the context below.
If the answer isn't in the context, say you don't know. Don't make stuff up.
Be concise."""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

def get_qa_chain(db_dir: str = "./index", k: int = 4):
    db = Chroma(
        persist_directory=db_dir,
        embedding_function=OpenAIEmbeddings(),
    )
    retriever = db.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # cheap and good enough

    def ask(question: str) -> dict:
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        # TODO: add re-ranking here, top-k is pretty dumb
        resp = (PROMPT | llm).invoke({"context": context, "question": question})
        return {
            "answer": resp.content,
            "sources": [d.metadata.get("source", "unknown") for d in docs],
        }

    return ask
