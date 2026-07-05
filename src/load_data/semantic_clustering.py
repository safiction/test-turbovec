import json
from pathlib import Path
from sklearn.datasets import fetch_20newsgroups

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "semantic_clustering"

MAX_TRAIN = 5000
MAX_TEST = 2000


def save_json(data, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path}")


def load_and_save():
    print("Loading 20newsgroups dataset...")

    train = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )

    test = fetch_20newsgroups(
        subset="test",
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )

    # Build category mapping from all target names
    target_names = train.target_names

    train_data = [
        {"text": text, "label": int(label), "category": target_names[label]}
        for text, label in zip(train.data[:MAX_TRAIN], train.target[:MAX_TRAIN])
        if text.strip()
    ]

    test_data = [
        {"text": text, "label": int(label), "category": target_names[label]}
        for text, label in zip(test.data[:MAX_TEST], test.target[:MAX_TEST])
        if text.strip()
    ]

    save_json(train_data, "train.json")
    save_json(test_data, "test.json")

    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Categories: {target_names}")


if __name__ == "__main__":
    load_and_save()
