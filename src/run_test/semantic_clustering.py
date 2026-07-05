"""
Semantic Clustering Benchmark for TurboVec

The benchmark compares exact k-NN (float32, per-query) against TurboVec quantized search (2-bit and 4-bit).
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
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from turbovec import TurboQuantIndex

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "semantic_clustering"
RESULTS_DIR = PROJECT_ROOT / "results" / "semantic_clustering"


def get_rss_mb():
    import psutil
    return psutil.Process().memory_info().rss / (1024 * 1024)


def load_json(filename):
    path = DATA_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_knn_accuracy(neighbor_labels, true_labels, k):
    """Majority vote among k neighbors."""
    correct = 0
    for neighbors, true in zip(neighbor_labels, true_labels):
        top_k = neighbors[:k]
        vote = Counter(top_k).most_common(1)[0][0]
        if vote == true:
            correct += 1
    return correct / len(true_labels)


def compute_mean_recall(neighbor_labels, true_labels, k):
    """Mean fraction of same-class neighbors in top-k."""
    recalls = []
    for neighbors, true in zip(neighbor_labels, true_labels):
        top_k = neighbors[:k]
        same_class = sum(1 for nl in top_k if nl == true)
        recalls.append(same_class / k)
    return np.mean(recalls)


def compute_mrr(neighbor_labels, true_labels):
    """Mean reciprocal rank of first same-class neighbor."""
    rr_sum = 0.0
    for neighbors, true in zip(neighbor_labels, true_labels):
        for rank, nl in enumerate(neighbors, start=1):
            if nl == true:
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(true_labels)


def compute_silhouette(embeddings, labels):
    """Silhouette score on cosine distance."""
    from sklearn.metrics.pairwise import cosine_distances

    n_samples = len(labels)
    n_labels = len(set(labels))
    if n_labels < 2 or n_labels > n_samples - 1:
        return None
    dists = cosine_distances(embeddings)
    return silhouette_score(dists, labels, metric="precomputed")


def run_baseline(
    train_embeddings,
    train_labels,
    test_embeddings,
    test_data,
    k,
    dim,
    run_id,
    save_every,
):
    """Run exact k-NN baseline (float32, per-query)."""
    print(f"[baseline] running exact k-NN (k={k})...")
    n_test = len(test_embeddings)

    csv_path = os.path.join(
        RESULTS_DIR, f"baseline_dim{dim}_k{k}_{run_id}.csv"
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fieldnames = [
        "query_idx",
        "query_text_preview",
        "true_label",
        "true_category",
        "knn_accuracy",
        "same_class_recall",
        "first_same_class_rank",
        "query_time_ms",
    ]

    all_neighbor_labels = []
    all_query_times = []

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, query_emb in enumerate(test_embeddings):
            t0 = time.perf_counter()
            similarities = train_embeddings @ query_emb
            top_k_idx = np.argpartition(similarities, -k)[-k:]
            top_k_idx = top_k_idx[np.argsort(-similarities[top_k_idx])]
            query_time_ms = (time.perf_counter() - t0) * 1000

            neighbor_labels = [train_labels[idx] for idx in top_k_idx]
            all_neighbor_labels.append(neighbor_labels)
            all_query_times.append(query_time_ms)

            top_k_labels = neighbor_labels[:k]
            vote = Counter(top_k_labels).most_common(1)[0][0]
            knn_acc = 1.0 if vote == test_data[i]["label"] else 0.0

            same_class_recall = sum(
                1 for nl in top_k_labels if nl == test_data[i]["label"]
            ) / k

            first_same_rank = None
            for rank, nl in enumerate(neighbor_labels, start=1):
                if nl == test_data[i]["label"]:
                    first_same_rank = rank
                    break

            writer.writerow(
                {
                    "query_idx": i,
                    "query_text_preview": test_data[i]["text"][:100].replace(
                        "\n", " "
                    ),
                    "true_label": test_data[i]["label"],
                    "true_category": test_data[i]["category"],
                    "knn_accuracy": knn_acc,
                    "same_class_recall": same_class_recall,
                    "first_same_class_rank": (
                        first_same_rank if first_same_rank else ""
                    ),
                    "query_time_ms": f"{query_time_ms:.4f}",
                }
            )

            if (i + 1) % save_every == 0 or i == n_test - 1:
                print(
                    f"  [baseline] processed {i + 1}/{n_test} queries, "
                    f"avg time={np.mean(all_query_times):.4f} ms"
                )

    knn_accuracy = compute_knn_accuracy(all_neighbor_labels, [d["label"] for d in test_data], k)
    mean_recall = compute_mean_recall(all_neighbor_labels, [d["label"] for d in test_data], k)
    mrr = compute_mrr(all_neighbor_labels, [d["label"] for d in test_data])
    silhouette = compute_silhouette(test_embeddings, [d["label"] for d in test_data])

    summary = {
        "knn_accuracy": knn_accuracy,
        "mean_same_class_recall_at_k": mean_recall,
        "mrr": mrr,
        "silhouette_score": silhouette,
        "avg_query_time_ms": np.mean(all_query_times),
        "total_time_ms": np.sum(all_query_times),
        "n_queries": n_test,
    }

    print(f"[baseline] results saved to {csv_path}")
    return csv_path, summary


def run_turbovec_eval(
    index,
    train_labels,
    model,
    test_data,
    test_embeddings,
    k,
    dim,
    bit_width,
    run_id,
    save_every,
):
    """Run TurboVec evaluation."""
    print(f"[turbovec] running search (k={k}, bit_width={bit_width})...")
    n_test = len(test_embeddings)

    csv_path = os.path.join(
        RESULTS_DIR, f"turbovec_bw{bit_width}_dim{dim}_k{k}_{run_id}.csv"
    )

    fieldnames = [
        "query_idx",
        "query_text_preview",
        "true_label",
        "true_category",
        "knn_accuracy",
        "same_class_recall",
        "first_same_class_rank",
        "query_time_ms",
    ]

    all_neighbor_labels = []
    all_query_times = []

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, query_emb in enumerate(test_embeddings):
            t0 = time.perf_counter()
            query_embedding = np.expand_dims(query_emb, axis=0).astype(np.float32)
            scores, indices = index.search(query_embedding, k=k)
            query_time_ms = (time.perf_counter() - t0) * 1000

            top_k_idx = indices[0]
            neighbor_labels = [train_labels[idx] for idx in top_k_idx]
            all_neighbor_labels.append(neighbor_labels)
            all_query_times.append(query_time_ms)

            top_k_labels = neighbor_labels[:k]
            vote = Counter(top_k_labels).most_common(1)[0][0]
            knn_acc = 1.0 if vote == test_data[i]["label"] else 0.0

            same_class_recall = sum(
                1 for nl in top_k_labels if nl == test_data[i]["label"]
            ) / k

            first_same_rank = None
            for rank, nl in enumerate(neighbor_labels, start=1):
                if nl == test_data[i]["label"]:
                    first_same_rank = rank
                    break

            writer.writerow(
                {
                    "query_idx": i,
                    "query_text_preview": test_data[i]["text"][:100].replace(
                        "\n", " "
                    ),
                    "true_label": test_data[i]["label"],
                    "true_category": test_data[i]["category"],
                    "knn_accuracy": knn_acc,
                    "same_class_recall": same_class_recall,
                    "first_same_class_rank": (
                        first_same_rank if first_same_rank else ""
                    ),
                    "query_time_ms": f"{query_time_ms:.4f}",
                }
            )

            if (i + 1) % save_every == 0 or i == n_test - 1:
                print(
                    f"  [turbovec] processed {i + 1}/{n_test} queries, "
                    f"avg time={np.mean(all_query_times):.4f} ms"
                )

    knn_accuracy = compute_knn_accuracy(all_neighbor_labels, [d["label"] for d in test_data], k)
    mean_recall = compute_mean_recall(all_neighbor_labels, [d["label"] for d in test_data], k)
    mrr = compute_mrr(all_neighbor_labels, [d["label"] for d in test_data])
    silhouette = compute_silhouette(test_embeddings, [d["label"] for d in test_data])

    summary = {
        "knn_accuracy": knn_accuracy,
        "mean_same_class_recall_at_k": mean_recall,
        "mrr": mrr,
        "silhouette_score": silhouette,
        "avg_query_time_ms": np.mean(all_query_times),
        "total_time_ms": np.sum(all_query_times),
        "n_queries": n_test,
    }

    print(f"[turbovec] results saved to {csv_path}")
    return csv_path, summary


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Clustering Benchmark for TurboVec"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=512,
        help="Embedding dimension (default: 512)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of neighbors for k-NN (default: 5)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Log progress every N queries (default: 10)",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run on a tiny subset for smoke testing",
    )
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run ID: {run_id}")
    print(f"Config: dim={args.dim}, k={args.k}")

    print("[load] reading train and test data...")
    train_data = load_json("train.json")
    test_data = load_json("test.json")

    if args.quick_test:
        train_data = train_data[:20]
        test_data = test_data[:5]
        print(
            f"[quick-test] limited to {len(train_data)} train, "
            f"{len(test_data)} test samples"
        )

    print(f"[load] {len(train_data)} train, {len(test_data)} test samples")

    train_texts = [s["text"] for s in train_data]
    train_labels = [s["label"] for s in train_data]
    test_texts = [s["text"] for s in test_data]

    print("[model] loading nomic-ai/nomic-embed-text-v1.5 ...")
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5", truncate_dim=args.dim
    )

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
        train_embeddings,
        train_labels,
        test_embeddings,
        test_data,
        args.k,
        args.dim,
        run_id,
        args.save_every,
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
            index,
            train_labels,
            model,
            test_data,
            test_embeddings,
            args.k,
            args.dim,
            bw,
            run_id,
            args.save_every,
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
    summary_path = os.path.join(
        RESULTS_DIR, f"summary_dim{args.dim}_k{args.k}_{run_id}.json"
    )

    memory_entries = {
        "baseline_embeddings": baseline_mb,
    }
    for tv in turbovec_results:
        bw = tv["bit_width"]
        n_train, dim = train_embeddings.shape
        memory_entries[f"turbovec_theoretical_bw{bw}"] = (
            n_train * dim * bw / 8
        ) / (1024 ** 2)

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