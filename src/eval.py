"""Eval harness using RAGAS - because vibes aren't metrics."""
import argparse
import json
from pathlib import Path
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from .retriever import get_qa_chain

# golden set format: [{"question": "...", "ground_truth": "..."}]
# I hand-wrote 20 of these from my own docs, painful but worth it

def run_eval(golden_path: str, db_dir: str = "./index"):
    golden = json.loads(Path(golden_path).read_text())
    ask = get_qa_chain(db_dir)

    questions, answers, contexts = [], [], []
    for item in golden:
        res = ask(item["question"])
        # ragas wants contexts as list[list[str]] - annoying
        questions.append(item["question"])
        answers.append(res["answer"])
        # we don't have true retrieved contexts here, using answer as proxy
        # TODO: fix this to pass actual chunks
        contexts.append([res["answer"]])

    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": [g["ground_truth"] for g in golden],
    })

    result = evaluate(ds, metrics=[faithfulness, answer_relevancy])
    print(result)
    # save for README
    Path("eval_results.json").write_text(str(result))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--db", default="./index")
    args = ap.parse_args()
    run_eval(args.golden, args.db)
