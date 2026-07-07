"""
Image Classification Benchmark for TurboVec
Compares exact k-NN (float32) vs TurboVec quantized search.

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
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from turbovec import TurboQuantIndex

MODEL_NAME = 'clip-vit-base-patch32'

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "image_classification"
RESULTS_DIR = PROJECT_ROOT / "results" / "image_classification"


def get_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_images(model, processor, image_paths, device, batch_size=32):
    """Encode images to CLIP embeddings, GPU if available."""
    all_embeddings = []
    n = len(image_paths)
    processed = 0
    for i in range(0, n, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        if hasattr(outputs, "pooler_output"):
            outputs = outputs.pooler_output
        embeddings = outputs.cpu().numpy()
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
        all_embeddings.append(embeddings.astype(np.float32))

        processed += len(batch_paths)
        if processed % 1000==0:
            print(f"[encode] {processed}/{n} images processed")

    return np.vstack(all_embeddings)


def majority_vote(labels):
    """Return the most common label."""
    return Counter(labels).most_common(1)[0][0]


def top_k_accuracy(neighbor_labels, true_labels, k):
    """Fraction of queries where true label is in top-k neighbors."""
    correct = 0
    for neighbors, true in zip(neighbor_labels, true_labels):
        if true in neighbors[:k]:
            correct += 1
    return correct / len(true_labels)


def per_class_accuracy(neighbor_labels, true_labels, num_classes, k):
    """Mean accuracy per class (using majority vote from top-k)."""
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


# Baseline: exact k-NN per-query
def run_baseline(train_embeddings, train_labels, test_embeddings, test_data, k, run_id, num_classes, save_every=1000):
    print(f"\n[baseline] running exact k-NN (k={k}) per-query...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"baseline_k{k}_{run_id}.csv")

    fieldnames = ["test_id", "true_label", "pred_label", "time_ms"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    n_test = len(test_data)
    rows_buffer = []
    total_time_ms = 0.0
    all_neighbor_labels = []
    predictions = []

    for i, sample in enumerate(test_data, start=1):
        query = test_embeddings[i - 1]

        start = time.time()
        scores = train_embeddings @ query
        top_indices = np.argsort(scores)[::-1][:k]
        neighbor_labels = [train_labels[j] for j in top_indices]
        pred = majority_vote(neighbor_labels)
        elapsed_ms = (time.time() - start) * 1000

        predictions.append(pred)
        all_neighbor_labels.append(neighbor_labels)

        row = {
            "test_id": i,
            "true_label": sample["label"],
            "pred_label": pred,
            "time_ms": elapsed_ms,
        }
        rows_buffer.append(row)
        total_time_ms += elapsed_ms

        is_flush_point = (i % save_every == 0) or (i == n_test)
        if is_flush_point:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerows(rows_buffer)

            batch_avg_time = sum(r["time_ms"] for r in rows_buffer) / len(rows_buffer)
            print(
                f"[baseline] {i}/{n_test} processed | "
                f"last batch avg time={batch_avg_time:.2f} ms | saved to {csv_path}"
            )
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


# TurboVec: quantized k-NN per-query
def run_turbovec_eval(index, train_labels, test_embeddings, test_data, k, bit_width, run_id, num_classes, save_every=1000):
    print(f"\n[turbovec] running k-NN (k={k}) for bit_width={bit_width}...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"turbovec_bw{bit_width}_k{k}_{run_id}.csv")

    fieldnames = ["test_id", "true_label", "pred_label", "time_ms"]

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

        predictions.append(pred)
        all_neighbor_labels.append(neighbor_labels)

        row = {
            "test_id": i,
            "true_label": sample["label"],
            "pred_label": pred,
            "time_ms": elapsed_ms,
        }
        rows_buffer.append(row)
        total_time_ms += elapsed_ms

        is_flush_point = (i % save_every == 0) or (i == n_test)
        if is_flush_point:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerows(rows_buffer)

            batch_avg_time = sum(r["time_ms"] for r in rows_buffer) / len(rows_buffer)
            print(
                f"[turbovec] {i}/{n_test} processed | "
                f"last batch avg time={batch_avg_time:.2f} ms | saved to {csv_path}"
            )
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

    print(f"[turbovec bw={bit_width}] done:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")

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

    num_classes = max(max(d["label"] for d in train_data), max(d["label"] for d in test_data)) + 1
    print(f"[load] num_classes={num_classes}")

    train_paths = [d["image_path"] for d in train_data]
    train_labels = [d["label"] for d in train_data]
    test_paths = [d["image_path"] for d in test_data]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[model] loading {MODEL_NAME} on {device} ...")
    mem_before_model = get_rss_mb()
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    mem_after_model = get_rss_mb()
    print(f"[model] loaded. process RSS delta={mem_after_model - mem_before_model:.2f} MB")

    print("[encode] embedding train images...")
    mem_before = get_rss_mb()
    train_embeddings = encode_images(model, processor, train_paths, device, batch_size=32)
    mem_after = get_rss_mb()
    print(
        f"[encode] done. shape={train_embeddings.shape}, "
        f"array size={train_embeddings.nbytes / (1024 ** 2):.2f} MB, "
        f"process RSS delta={mem_after - mem_before:.2f} MB"
    )

    print("[encode] embedding test images...")
    test_embeddings = encode_images(model, processor, test_paths, device, batch_size=32)

    baseline_csv_path, baseline_summary, _ = run_baseline(
        train_embeddings, train_labels, test_embeddings, test_data,
        args.k, run_id, num_classes,
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
            index, train_labels, test_embeddings, test_data,
            args.k, bw, run_id, num_classes,
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