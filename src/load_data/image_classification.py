"""
Load Oxford-IIIT Pet dataset and save train/test metadata as JSON.
"""
import json
from pathlib import Path
from torchvision.datasets import OxfordIIITPet

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "image_classification"


def save_json(data, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path}")


def load_and_save():
    print("Loading Oxford-IIIT Pet dataset...")

    train = OxfordIIITPet(
        root=OUTPUT_DIR,
        split="trainval",
        target_types="category",
        download=True,
    )

    test = OxfordIIITPet(
        root=OUTPUT_DIR,
        split="test",
        target_types="category",
        download=True,
    )

    label_names = train.classes  # list of breed names

    train_data = [
        {
            "image_path": str(Path(img_path).resolve()),
            "label": int(label),
            "breed": label_names[label],
        }
        for img_path, label in zip(train._images, train._labels)
    ]

    test_data = [
        {
            "image_path": str(Path(img_path).resolve()),
            "label": int(label),
            "breed": label_names[label],
        }
        for img_path, label in zip(test._images, test._labels)
    ]

    save_json(train_data, "train.json")
    save_json(test_data, "test.json")

    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Classes: {len(label_names)}")
    print(f"Class names: {label_names}")


if __name__ == "__main__":
    load_and_save()
