import json
import os
from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "image_classification"


def save_json(data, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path}")


dataset_train = load_dataset('tanganke/sun397')['train']
dataset_test = load_dataset('tanganke/sun397')['test']

print(f"Train size: {len(dataset_train)} images")
print(f"Test size: {len(dataset_test)} images")

train_data = []
for i, sample in enumerate(dataset_train):
    image = sample['image']
    # Save image to disk so run_test can load it later
    img_path = OUTPUT_DIR / "train_images" / f"{i}.jpg"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(img_path)
    train_data.append({"image_path": str(img_path), "label": int(sample['label'])})
    if i%1000:
        print(i, "images saved")

test_data = []
for i, sample in enumerate(dataset_test):
    image = sample['image']
    img_path = OUTPUT_DIR / "test_images" / f"{i}.jpg"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(img_path)
    test_data.append({"image_path": str(img_path), "label": int(sample['label'])})
    if i%1000:
        print(i, "images saved")

save_json(train_data, "train.json")
save_json(test_data, "test.json")

print(f"Train samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")
print(f"Num classes: {max(d['label'] for d in train_data) + 1}")