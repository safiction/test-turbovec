"""
Benchmark: k-NN anomaly detection via exact (baseline) search vs TurboVec.

Compares two variants:
    - raw        : original V1-V28 features as-is
    - normalized : L2-normalized features (unit vectors, as TurboQuant expects)

For each variant we run:
    - Baseline : exact k-NN per-query on float32 embeddings
    - TurboVec : k-NN per-query via TurboQuantIndex (bit_width=2, 4)

Usage:
    py anomaly_detection.py --k 5
    py anomaly_detection.py --k 10 --quick-test
"""

import os
import time
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil

from turbovec import TurboQuantIndex
from sklearn.metrics import average_precision_score, precision_recall_curve, auc

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "anomaly_detection"
RESULTS_DIR = "results/anomaly_detection"

def get_rss_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

def normalize_l2(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row to unit length."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def pad_to_multiple_of_8(vectors: np.ndarray) -> np.ndarray:
    """Pad vectors with zeros so dim is a multiple of 8 (TurboQuant requirement)."""
    dim = vectors.shape[1]
    pad_dim = (8 - dim % 8) % 8
    if pad_dim == 0:
        return vectors
    padding = np.zeros((vectors.shape[0], pad_dim), dtype=vectors.dtype)
    return np.concatenate([vectors, padding], axis=1)


def compute_anomaly_scores(neighbor_scores: np.ndarray) -> np.ndarray:
    """
    Convert k-NN similarity scores to anomaly scores.
    Higher score = more anomalous.
    """
    return 1.0 - np.mean(neighbor_scores, axis=1)


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """
    Compute AP and AUC-PR.
    """
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    pr_auc = auc(recall, precision)
    return {
        "average_precision": float(ap),
        "auc_pr": float(pr_auc),
    }

def run_baseline(X_train, X_test, y_test, k, variant, run_id, save_every=1000):
    print(f"\n[baseline | {variant}] running exact k-NN (k={k}) per-query...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(
        RESULTS_DIR, f"baseline_{variant}_k{k}_{run_id}.csv"
    )

    fieldnames = ["test_id", "true_label", "anomaly_score", "time_ms"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    n_test = len(X_test)
    rows_buffer = []
    total_time_ms = 0.0
    all_scores = []

    for i, query in enumerate(X_test, start=1):
        start = time.time()
        scores = X_train @ query
        top_k = np.sort(scores)[::-1][:k]
        anomaly_score = 1.0 - np.mean(top_k)
        elapsed_ms = (time.time() - start) * 1000

        all_scores.append(anomaly_score)

        row = {
            "test_id": i,
            "true_label": int(y_test[i - 1]),
            "anomaly_score": anomaly_score,
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
                f"[baseline | {variant}] {i}/{n_test} processed | "
                f"last batch avg time={batch_avg_time:.2f} ms | saved to {csv_path}"
            )
            rows_buffer = []

    all_scores = np.array(all_scores)
    metrics = compute_metrics(y_test, all_scores)

    summary = {
        "total_time_sec": total_time_ms / 1000,
        "avg_time_per_query_ms": total_time_ms / n_test,
        **metrics,
        "memory_embeddings_mb": X_train.nbytes / (1024 ** 2),
    }

    print(f"[baseline | {variant}] done:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")

    return csv_path, summary


def run_turbovec_eval(
    index, X_test, y_test, k, bit_width, variant, run_id, save_every=1000
):
    print(
        f"\n[turbovec | {variant}] running k-NN (k={k}) "
        f"for bit_width={bit_width}..."
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(
        RESULTS_DIR, f"turbovec_{variant}_bw{bit_width}_k{k}_{run_id}.csv"
    )

    fieldnames = ["test_id", "true_label", "anomaly_score", "time_ms"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    n_test = len(X_test)
    rows_buffer = []
    total_time_ms = 0.0
    all_scores = []

    for i, query in enumerate(X_test, start=1):
        query_embedding = np.expand_dims(query, axis=0).astype(np.float32)

        start = time.time()
        scores, _ = index.search(query_embedding, k=k)
        top_k = scores[0]
        anomaly_score = 1.0 - np.mean(top_k)
        elapsed_ms = (time.time() - start) * 1000

        all_scores.append(anomaly_score)

        row = {
            "test_id": i,
            "true_label": int(y_test[i - 1]),
            "anomaly_score": anomaly_score,
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
                f"[turbovec | {variant}] {i}/{n_test} processed | "
                f"last batch avg time={batch_avg_time:.2f} ms | saved to {csv_path}"
            )
            rows_buffer = []

    all_scores = np.array(all_scores)
    metrics = compute_metrics(y_test, all_scores)

    summary = {
        "total_time_sec": total_time_ms / 1000,
        "avg_time_per_query_ms": total_time_ms / n_test,
        **metrics,
    }

    return csv_path, summary

def run_variant(X_train, X_test, y_test, k, variant, run_id, save_every):
    """Run baseline + TurboVec for a single variant (raw or normalized)."""
    print(f"\n{'='*60}")
    print(f"VARIANT: {variant}")
    print(f"{'='*60}")

    # --- Baseline ---
    baseline_csv, baseline_summary = run_baseline(
        X_train, X_test, y_test, k, variant, run_id, save_every
    )
    baseline_mb = X_train.nbytes / (1024 ** 2)

    # --- TurboVec ---
    turbovec_results = []
    BIT_WIDTHS = [2, 4]

    for bw in BIT_WIDTHS:
        print(f"\n[turbovec | {variant}] building index for bit_width={bw}...")
        mem_before_index = get_rss_mb()
        index = TurboQuantIndex(dim=X_train.shape[1], bit_width=bw)
        index.add(X_train)
        mem_after_index = get_rss_mb()

        turbovec_rss_delta_mb = mem_after_index - mem_before_index
        n_train, dim = X_train.shape
        theoretical_turbovec_mb = (n_train * dim * bw / 8) / (1024 ** 2)

        print(
            f"[turbovec | {variant}] index built. "
            f"process RSS delta={turbovec_rss_delta_mb:.2f} MB, "
            f"theoretical size={theoretical_turbovec_mb:.2f} MB, "
            f"vs baseline float32={baseline_mb:.2f} MB "
            f"(~{baseline_mb / theoretical_turbovec_mb:.1f}x smaller in theory)"
        )

        csv_path, turbovec_summary = run_turbovec_eval(
            index, X_test, y_test, k, bw, variant, run_id, save_every
        )
        turbovec_summary["bit_width"] = bw
        turbovec_results.append(turbovec_summary)

        print(f"\nTurboVec | {variant} bit_width={bw} results:")
        for key, value in turbovec_summary.items():
            print(f"    {key}: {value}")

    return {
        "baseline": baseline_summary,
        "turbovec": turbovec_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="k-NN anomaly detection: baseline vs TurboVec (raw + normalized)."
    )
    parser.add_argument("--k", type=int, default=5, help="number of neighbors")
    parser.add_argument(
        "--save-every", type=int, default=1000, help="flush results to CSV every N samples"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="run on reduced test set for a smoke test",
    )
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== run_id={run_id} k={args.k} ===")

    print("[load] reading train and test data...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    print(f"[load] {len(X_train)} train, {len(X_test)} test samples")
    print(f"[load] anomalies in test: {y_test.sum()} / {len(y_test)} "
          f"({100 * y_test.mean():.2f}%)")

    if args.quick_test:
        X_test = X_test[:1000]
        y_test = y_test[:1000]
        print(f"[quick-test] limited to {len(X_test)} test samples")

    X_train_padded = pad_to_multiple_of_8(X_train)
    X_test_padded = pad_to_multiple_of_8(X_test)
    print(f"[pad] dim {X_train.shape[1]} -> {X_train_padded.shape[1]} (padded to multiple of 8)")

    raw_results = run_variant(
        X_train_padded, X_test_padded, y_test, args.k, "raw", run_id, args.save_every
    )

    # Normalized variant
    X_train_norm = normalize_l2(X_train)
    X_test_norm = normalize_l2(X_test)
    X_train_norm_padded = pad_to_multiple_of_8(X_train_norm)
    X_test_norm_padded = pad_to_multiple_of_8(X_test_norm)

    norm_results = run_variant(
        X_train_norm_padded, X_test_norm_padded, y_test, args.k, "normalized", run_id, args.save_every
    )

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print("\n--- RAW ---")
    print("Baseline (exact k-NN, float32, per-query):")
    for key, value in raw_results["baseline"].items():
        print(f"    {key}: {value}")
    print("\nTurboVec results by bit_width:")
    for tv in raw_results["turbovec"]:
        print(f"  bit_width={tv['bit_width']}:")
        for key, value in tv.items():
            if key == "bit_width":
                continue
            print(f"      {key}: {value}")

    print("\n--- NORMALIZED ---")
    print("Baseline (exact k-NN, float32, per-query):")
    for key, value in norm_results["baseline"].items():
        print(f"    {key}: {value}")
    print("\nTurboVec results by bit_width:")
    for tv in norm_results["turbovec"]:
        print(f"  bit_width={tv['bit_width']}:")
        for key, value in tv.items():
            if key == "bit_width":
                continue
            print(f"      {key}: {value}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, f"summary_k{args.k}_{run_id}.json")

    n_train, dim = X_train.shape
    memory_entries = {
        "baseline_embeddings": X_train.nbytes / (1024 ** 2),
    }
    for bw in [2, 4]:
        memory_entries[f"turbovec_theoretical_bw{bw}"] = (
            n_train * dim * bw / 8
        ) / (1024 ** 2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "k": args.k,
                "n_train": int(n_train),
                "n_test": int(len(X_test)),
                "dim": int(dim),
                "anomaly_rate_test": float(y_test.mean()),
                "variants": {
                    "raw": raw_results,
                    "normalized": norm_results,
                },
                "memory_mb": memory_entries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nRun summary: {summary_path}")

if __name__ == "__main__":
    main()
