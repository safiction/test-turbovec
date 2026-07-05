import json
from pathlib import Path
from datasets import load_dataset

MAX_QUESTIONS = {"train": 5000,"validation": 2000}
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "rag_search"

def save_json(data, filename):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    path = OUTPUT_DIR / filename
    with open(path,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)
    print(f"Saved: {path}")


def load_ds(split_name: str):
    """
    Loads SberQuAD split and converts it into:
    contexts:
    {
        context_id,
        text
    }

    questions:
    {
        question_id,
        question_text,
        context_id
    }
    """

    ds = load_dataset("kuznetsoffandrey/sberquad")[split_name]

    # context text -> generated id
    context_to_id = {}

    contexts = []
    questions = []

    context_counter = 1
    question_counter = 1

    for sample in ds:

        context_text = sample["context"]

        if context_text not in context_to_id:
            context_id = f"context_{context_counter}"
            context_to_id[context_text] = context_id


            contexts.append(
                {
                    "context_id": context_id,
                    "text": context_text
                }
            )

            context_counter += 1


        context_id = context_to_id[context_text]

        if question_counter <= MAX_QUESTIONS[split_name]:
            questions.append(
                {
                    "question_id": f"question_{question_counter}",
                    "question_text": sample["question"],
                    "context_id": context_id
                }
            )
            question_counter += 1

    save_json(
        contexts,
        f"{split_name}_contexts.json"
    )

    save_json(
        questions,
        f"{split_name}_questions.json"
    )

    print()
    print(f"Split: {split_name}")
    print(f"Samples: {len(ds)}")
    print(f"Unique contexts: {len(contexts)}")
    print(f"Questions: {len(questions)}")
    print("-" * 50)


if __name__ == "__main__":
    load_ds("train")
    load_ds("validation")