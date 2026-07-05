import json
from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "classification"

MAX_TRAIN = 10_000
MAX_TEST = 1000


def save_json(data, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path}")


def load_and_save():
    print("Loading IMDB dataset...")
    dataset = load_dataset("stanfordnlp/imdb")

    train = dataset["train"]
    test = dataset["test"]

    train_data = [
        {"text": sample["text"], "label": sample["label"]}
        for sample in train.select(range(min(MAX_TRAIN, len(train))))
    ]

    test_data = [
        {"text": sample["text"], "label": sample["label"]}
        for sample in test.select(range(min(MAX_TEST, len(test))))
    ]

    save_json(train_data, "train.json")
    save_json(test_data, "test.json")

    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")


if __name__ == "__main__":
    load_and_save()
