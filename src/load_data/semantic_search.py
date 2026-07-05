import json
from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "semantic_search"

MAX_PASSAGES = 40_000
MAX_QUERIES = 10_000


def save_json(data, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path}")


def load_and_save():
    print("Loading MS MARCO passage ranking dataset...")

    passages_ds = load_dataset("microsoft/ms_marco", "v1.1", split="train")

    # Collect unique passages
    passage_to_id = {}
    corpus = []

    for sample in passages_ds:
        for passage_text in sample["passages"]["passage_text"]:
            if passage_text not in passage_to_id:
                doc_id = f"doc_{len(corpus)}"
                passage_to_id[passage_text] = doc_id
                corpus.append({"doc_id": doc_id, "text": passage_text})

            if len(corpus) >= MAX_PASSAGES:
                break
        if len(corpus) >= MAX_PASSAGES:
            break

    queries = []
    qrels = {}

    query_count = 0
    for sample in passages_ds:
        query_id = sample["query_id"]
        query_text = sample["query"]

        # Get relevant passages (is_selected == 1)
        relevant_passages = []
        for is_selected, passage_text in zip(
            sample["passages"]["is_selected"], sample["passages"]["passage_text"]
        ):
            if is_selected == 1 and passage_text in passage_to_id:
                relevant_passages.append(passage_to_id[passage_text])

        # Only include queries that have at least one relevant passage in our corpus
        if relevant_passages:
            queries.append({
                "query_id": str(query_id),
                "query_text": query_text,
            })
            qrels[str(query_id)] = relevant_passages
            query_count += 1

            if query_count >= MAX_QUERIES:
                break

    save_json(corpus, "corpus.json")
    save_json(queries, "queries.json")
    save_json(qrels, "qrels.json")

    print()
    print(f"Passages (corpus): {len(corpus)}")
    print(f"Queries: {len(queries)}")
    print(f"Qrels (queries with relevance): {len(qrels)}")

if __name__ == "__main__":
    load_and_save()