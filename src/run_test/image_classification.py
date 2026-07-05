"""
Image Classification Benchmark for TurboVec
Compares exact k-NN (float32) vs TurboVec quantized search.

Metrics:
    - Top-1 Accuracy
    - Top-5 Accuracy (if k >= 5)
    - F1-macro, F1-weighted
    - Per-class accuracy (mean across all classes)
    - Query time, memory usage

Usage:
    py src/run_test/image_classification.py --k 5
    py src/run_test/image_classification.py --k 5 --quick-test
"""

import argparse
import csv
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import torch
import timm
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torchvision import transforms

from turbovec import TurboQuantIndex

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "image_classification"
RESULTS_DIR = PROJECT_ROOT / "results" / "image_classification"

MODEL_NAME = "resnet50.a1_in1k"
IMAGE_SIZE = 224

def get_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def majority_vote(labels):
    return Counter(labels).most_common(1)[0][0]


def top_k_accuracy(neighbor_labels, true_labels, k):
    """Fraction of queries where true label is in top-k neighbors."""
    correct = 0
    for neighbors, true in zip(neighbor_labels, true_labels):
        if true in neighbors[:k]:
            correct += 1
    return correct / len(true_labels)


def per_class_accuracy(neighbor_labels, true_labels, num_classes, k):
    """Mean accuracy per class."""
    class_correct = {i: 0 for i in range(num_classes)}
    class_total = {i: 0 for i in range(num_classes)}

    for neighbors, true in zip(neighbor_labels, true_labels):
        top_k = neighbors[:k]
        pred = Counter(top_k).most_common(1)[0][0]
        class_total[true] += 1
        if pred == true:
            class_correct[true] += 1

    accuracies = []
    for i in range(num_classes):
        if class_total[i] > 0:
            accuracies.append(class_correct[i] / class_total[i])
    return np.mean(accuracies) if accuracies else 0.0

def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_image(path):
    return Image.open(path).convert("RGB")


def create_model():
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    model.eval()
    return model


def encode_images(model, image_paths, batch_size=32, device="cpu"):
    """Encode images to embeddings using ResNet50."""
    transform = get_transform()
    model = model.to(device)
    embeddings = []

    total = len(image_paths)
    for i in range(0, total, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = torch.stack([transform(load_image(p)) for p in batch_paths])
        images = images.to(device)

        with torch.no_grad():
            batch_emb = model(images)

        batch_emb = batch_emb.cpu().numpy()
        # Normalize
        batch_emb /= np.linalg.norm(batch_emb, axis=1, keepdims=True)
        embeddings.append(batch_emb)

        processed = min(i + batch_size, total)
        if processed % 1000 == 0 or processed == total:
            print(f"[encode] {processed}/{total} images processed...")

    return np.vstack(embeddings).astype(np.float32)


# Baseline: exact k-NN
def run_baseline(train_embeddings, train_labels, test_embeddings, test_data, k, label_to_breed, run_id, num_classes, save_every=20):
    print(f"\n[baseline] running exact k-NN (k={k}) per-query...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"baseline_k{k}_{run_id}.csv")

    fieldnames = ["test_id", "image_path", "true_label", "true_breed", "pred_label", "pred_breed", "time_ms"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    n_test = len(test_data)
    rows_buffer = []
    total_time_ms = 0.0
    all_neighbor_labels = []
    predictions = []

    for i, sample in enumerate(test_data, start=1):
        query = test_embeddings[i - 1]
        if query.ndim == 2:
            query = query[0]

        start = time.time()
        scores = train_embeddings @ query
        top_indices = np.argsort(scores)[::-1][:k]
        neighbor_labels = [train_labels[j] for j in top_indices]
        pred = majority_vote(neighbor_labels)
        elapsed_ms = (time.time() - start) * 1000

        all_neighbor_labels.append(neighbor_labels)
        predictions.append(pred)

        row = {
            "test_id": i,
            "image_path": sample["image_path"],
            "true_label": sample["label"],
            "true_breed": sample["breed"],
            "pred_label": pred,
            "pred_breed": label_to_breed.get(pred, ""),
            "time_ms": elapsed_ms,
        }
        rows_buffer.append(row)
        total_time_ms += elapsed_ms

        is_flush_point = (i % save_every == 0) or (i == n_test)
        if is_flush_point:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerows(rows_buffer)

            batch_avg_time = sum(r["time_ms"] for r in rows_buffer) / len(rows_buffer)
            print(f"[baseline] {i}/{n_test} processed | last batch avg time={batch_avg_time:.2f} ms")
            rows_buffer = []

    true_labels = [s["label"] for s in test_data]

    summary = {
        "total_time_sec": total_time_ms / 1000,
        "avg_time_per_query_ms": total_time_ms / n_test,
        "top1_accuracy": accuracy_score(true_labels, predictions),
        "f1_macro": f1_score(true_labels, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(true_labels, predictions, average="weighted", zero_division=0),
        "per_class_accuracy": per_class_accuracy(all_neighbor_labels, true_labels, num_classes, k),
        "memory_embeddings_mb": train_embeddings.nbytes / (1024 ** 2),
    }

    if k >= 5:
        summary["top5_accuracy"] = top_k_accuracy(all_neighbor_labels, true_labels, 5)

    print("[baseline] done:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")

    return csv_path, summary, all_neighbor_labels


# TurboVec: quantized k-NN
def run_turbovec_eval(index, train_labels, test_data, test_embeddings, k, bit_width, label_to_breed, run_id, num_classes, save_every=20):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"turbovec_bw{bit_width}_k{k}_{run_id}.csv")

    fieldnames = ["test_id", "image_path", "true_label", "true_breed", "pred_label", "pred_breed", "time_ms"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    n_test = len(test_data)
    rows_buffer = []
    total_time_ms = 0.0
    all_neighbor_labels = []
    predictions = []

    for i, sample in enumerate(test_data, start=1):
        query_embedding = np.expand_dims(test_embeddings[i - 1], axis=0).astype(np.float32)

        start = time.time()
        scores, indices = index.search(query_embedding, k=k)
        neighbor_labels = [train_labels[j] for j in indices[0]]
        pred = majority_vote(neighbor_labels)
        elapsed_ms = (time.time() - start) * 1000

        all_neighbor_labels.append(neighbor_labels)
        predictions.append(pred)

        row = {
            "test_id": i,
            "image_path": sample["image_path"],
            "true_label": sample["label"],
            "true_breed": sample["breed"],
            "pred_label": pred,
            "pred_breed": label_to_breed.get(pred, ""),
            "time_ms": elapsed_ms,
        }
        rows_buffer.append(row)
        total_time_ms += elapsed_ms

        is_flush_point = (i % save_every == 0) or (i == n_test)
        if is_flush_point:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerows(rows_buffer)

            batch_avg_time = sum(r["time_ms"] for r in rows_buffer) / len(rows_buffer)
            print(f"[turbovec] {i}/{n_test} processed | last batch avg time={batch_avg_time:.2f} ms")
            rows_buffer = []

    true_labels = [s["label"] for s in test_data]

    summary = {
        "total_time_sec": total_time_ms / 1000,
        "avg_time_per_query_ms": total_time_ms / n_test,
        "top1_accuracy": accuracy_score(true_labels, predictions),
        "f1_macro": f1_score(true_labels, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(true_labels, predictions, average="weighted", zero_division=0),
        "per_class_accuracy": per_class_accuracy(all_neighbor_labels, true_labels, num_classes, k),
    }

    if k >= 5:
        summary["top5_accuracy"] = top_k_accuracy(all_neighbor_labels, true_labels, 5)

    return csv_path, summary, all_neighbor_labels

def main():
    parser = argparse.ArgumentParser(description="Image Classification Benchmark for TurboVec")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors for k-NN")
    parser.add_argument("--quick-test", action="store_true", help="Run on a tiny subset for smoke testing")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== run_id={run_id} k={args.k} ===")

    print("[load] reading train and test data...")
    train_data = load_json(DATA_DIR / "train.json")
    test_data = load_json(DATA_DIR / "test.json")

    if args.quick_test:
        train_data = train_data[:50]
        test_data = test_data[:10]
        print(f"[quick-test] limited to {len(train_data)} train, {len(test_data)} test samples")

    print(f"[load] {len(train_data)} train, {len(test_data)} test samples")

    num_classes = len({s["label"] for s in train_data})
    print(f"[load] num_classes={num_classes}")

    train_labels = [s["label"] for s in train_data]
    train_paths = [s["image_path"] for s in train_data]
    test_paths = [s["image_path"] for s in test_data]

    # predictions are always a label drawn from train_labels, so the
    # label->breed mapping must be built from train_data (not test_data, which may not cover every class)
    label_to_breed = {s["label"]: s["breed"] for s in train_data}

    print(f"[model] loading {MODEL_NAME} ...")
    mem_before_model = get_rss_mb()
    model = create_model()
    mem_after_model = get_rss_mb()
    print(f"[model] loaded. process RSS delta={mem_after_model - mem_before_model:.2f} MB")

    print("[encode] embedding train images...")
    mem_before = get_rss_mb()
    train_embeddings = encode_images(model, train_paths)
    mem_after = get_rss_mb()
    print(
        f"[encode] done. shape={train_embeddings.shape}, "
        f"array size={train_embeddings.nbytes / (1024 ** 2):.2f} MB, "
        f"process RSS delta={mem_after - mem_before:.2f} MB"
    )

    print("[encode] embedding test images...")
    test_embeddings = encode_images(model, test_paths)

    baseline_csv_path, baseline_summary, _ = run_baseline(
        train_embeddings, train_labels, test_embeddings, test_data,
        args.k, label_to_breed, run_id, num_classes,
    )

    baseline_mb = train_embeddings.nbytes / (1024 ** 2)
    turbovec_results = []

    BIT_WIDTHS = [2, 4]
    for bw in BIT_WIDTHS:
        print(f"\n[turbovec] building index for bit_width={bw}...")
        mem_before_index = get_rss_mb()
        index = TurboQuantIndex(dim=train_embeddings.shape[1], bit_width=bw)
        index.add(train_embeddings)
        mem_after_index = get_rss_mb()

        turbovec_rss_delta_mb = mem_after_index - mem_before_index
        n_train, dim = train_embeddings.shape
        theoretical_turbovec_mb = (n_train * dim * bw / 8) / (1024 ** 2)

        print(
            f"[turbovec] index built. "
            f"process RSS delta={turbovec_rss_delta_mb:.2f} MB, "
            f"theoretical size={theoretical_turbovec_mb:.2f} MB, "
            f"vs baseline float32={baseline_mb:.2f} MB "
            f"(~{baseline_mb / theoretical_turbovec_mb:.1f}x smaller in theory)"
        )

        print(f"\n[turbovec] running k-NN (k={args.k}) for bit_width={bw}...")
        csv_path, turbovec_summary, _ = run_turbovec_eval(
            index, train_labels, test_data, test_embeddings,
            args.k, bw, label_to_breed, run_id, num_classes,
        )

        turbovec_summary["bit_width"] = bw
        turbovec_results.append(turbovec_summary)

        print(f"\nTurboVec bit_width={bw} results:")
        for key, value in turbovec_summary.items():
            print(f"    {key}: {value}")

    print("\n=== SUMMARY ===")
    print("Baseline (exact k-NN, float32, per-query):")
    for key, value in baseline_summary.items():
        print(f"    {key}: {value}")

    print("\nTurboVec results by bit_width:")
    for tv in turbovec_results:
        print(f"  bit_width={tv['bit_width']}:")
        for key, value in tv.items():
            if key == "bit_width":
                continue
            print(f"      {key}: {value}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, f"summary_k{args.k}_{run_id}.json")

    memory_entries = {
        "baseline_embeddings": baseline_mb,
        "model_rss_delta_mb": mem_after_model - mem_before_model,
    }
    for tv in turbovec_results:
        bw = tv["bit_width"]
        n_train, dim = train_embeddings.shape
        memory_entries[f"turbovec_theoretical_bw{bw}"] = (n_train * dim * bw / 8) / (1024 ** 2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "k": args.k,
                "n_train": len(train_data),
                "n_test": len(test_data),
                "num_classes": num_classes,
                "model": MODEL_NAME,
                "baseline": baseline_summary,
                "turbovec": turbovec_results,
                "memory_mb": memory_entries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nBaseline results: {baseline_csv_path}")
    print(f"Run summary: {summary_path}")


if __name__ == "__main__":
    main()