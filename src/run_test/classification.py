"""
Benchmark: k-NN classification via exact (baseline) search vs TurboVec.

Usage:
    py classification.py --dim 384 --k 5
    py classification.py --dim 512 --k 10 --quick-test
"""

import os
import time
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import psutil

from turbovec import TurboQuantIndex
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "classification"

RESULTS_DIR = "results/classification"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_rss_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

def majority_vote(labels):
    """Return the most common label."""
    return Counter(labels).most_common(1)[0][0]

# Baseline: exact k-NN per-query
def run_baseline(train_embeddings, train_labels, test_embeddings, test_data, k, dim, run_id, save_every=20):
    print(f"\n[baseline] running exact k-NN (k={k}) per-query...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"baseline_dim{dim}_k{k}_{run_id}.csv")

    fieldnames = ["test_id", "text", "true_label", "pred_label", "time_ms"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    n_test = len(test_data)
    rows_buffer = []
    total_time_ms = 0.0
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

        predictions.append(pred)

        row = {
            "test_id": i,
            "text": sample["text"][:200] + "..." if len(sample["text"]) > 200 else sample["text"],
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
        "accuracy": accuracy_score(true_labels, predictions),
        "f1_macro": f1_score(true_labels, predictions, average="macro"),
        "f1_weighted": f1_score(true_labels, predictions, average="weighted"),
        "memory_embeddings_mb": train_embeddings.nbytes / (1024 ** 2),
    }

    print("[baseline] done:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")

    return csv_path, summary

# TurboVec: k-NN per-query via quantized index
def run_turbovec_eval(index, train_labels, model, test_data, test_embeddings, k, dim, bit_width, run_id, save_every=20):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"turbovec_dim{dim}_bw{bit_width}_k{k}_{run_id}.csv")

    fieldnames = ["test_id", "text", "true_label", "pred_label", "time_ms"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    n_test = len(test_data)
    rows_buffer = []
    total_time_ms = 0.0
    predictions = []

    for i, sample in enumerate(test_data, start=1):
        query_embedding = np.expand_dims(test_embeddings[i - 1], axis=0).astype(np.float32)

        start = time.time()
        scores, indices = index.search(query_embedding, k=k)
        neighbor_labels = [train_labels[j] for j in indices[0]]
        pred = majority_vote(neighbor_labels)
        elapsed_ms = (time.time() - start) * 1000

        predictions.append(pred)

        row = {
            "test_id": i,
            "text": sample["text"][:200] + "..." if len(sample["text"]) > 200 else sample["text"],
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
        "accuracy": accuracy_score(true_labels, predictions),
        "f1_macro": f1_score(true_labels, predictions, average="macro"),
        "f1_weighted": f1_score(true_labels, predictions, average="weighted"),
    }

    return csv_path, summary


def main():
    parser = argparse.ArgumentParser(description="k-NN classification: baseline vs TurboVec.")
    parser.add_argument("--dim", type=int, default=384, help="embedding dimension (truncate_dim)")
    parser.add_argument("--k", type=int, default=5, help="number of neighbors for k-NN")
    parser.add_argument("--save-every", type=int, default=20, help="flush results to CSV every N samples")
    parser.add_argument("--quick-test", action="store_true", help="run on 100 train / 10 test samples for a smoke test")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== run_id={run_id} dim={args.dim} k={args.k} ===")

    print("[load] reading train and test data...")
    train_data = load_json(DATA_DIR / "train.json")
    test_data = load_json(DATA_DIR / "test.json")

    if args.quick_test:
        train_data = train_data[:100]
        test_data = test_data[:10]
        print(f"[quick-test] limited to {len(train_data)} train, {len(test_data)} test samples")

    print(f"[load] {len(train_data)} train, {len(test_data)} test samples")

    train_texts = [s["text"] for s in train_data]
    train_labels = [s["label"] for s in train_data]
    test_texts = [s["text"] for s in test_data]

    print("[model] loading nomic-ai/nomic-embed-text-v1.5 ...")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", truncate_dim=args.dim)

    print("[encode] embedding train texts...")
    mem_before = get_rss_mb()
    train_embeddings = model.encode(
        train_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    ).astype(np.float32)
    mem_after = get_rss_mb()
    print(
        f"[encode] done. shape={train_embeddings.shape}, "
        f"array size={train_embeddings.nbytes / (1024 ** 2):.2f} MB, "
        f"process RSS delta={mem_after - mem_before:.2f} MB"
    )

    print("[encode] embedding test texts...")
    test_embeddings = model.encode(
        test_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    ).astype(np.float32)

    baseline_csv_path, baseline_summary = run_baseline(
        train_embeddings, train_labels, test_embeddings, test_data,
        args.k, args.dim, run_id, args.save_every,
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
        csv_path, turbovec_summary = run_turbovec_eval(
            index, train_labels, model, test_data, test_embeddings,
            args.k, args.dim, bw, run_id, args.save_every,
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
    summary_path = os.path.join(RESULTS_DIR, f"summary_dim{args.dim}_k{args.k}_{run_id}.json")

    memory_entries = {
        "baseline_embeddings": baseline_mb,
    }
    for tv in turbovec_results:
        bw = tv["bit_width"]
        n_train, dim = train_embeddings.shape
        memory_entries[f"turbovec_theoretical_bw{bw}"] = (n_train * dim * bw / 8) / (1024 ** 2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "dim": args.dim,
                "k": args.k,
                "n_train": len(train_data),
                "n_test": len(test_data),
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
