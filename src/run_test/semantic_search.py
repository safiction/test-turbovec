"""
Semantic Search Benchmark for TurboVec (MS MARCO)

Usage:
    py semantic_search.py
    py semantic_search.py --limit 200   # quick dry run
    py semantic_search.py --quick-test   # smoke test
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
from sentence_transformers import SentenceTransformer

from turbovec import TurboQuantIndex

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "semantic_search"
RESULTS_DIR = PROJECT_ROOT / "results" / "semantic_search"

MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
BIT_WIDTHS = [2, 4]
K_VALUES = [1, 5, 10]


def get_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def load_json(filename):
    path = DATA_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Metrics
def dcg_at_k(relevances, k):
    """Discounted Cumulative Gain."""
    relevances = np.array(relevances)[:k]
    if len(relevances) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(relevances / discounts))


def ndcg_at_k(predicted_ids, target_ids, k):
    """Normalized DCG@k."""
    relevances = [1.0 if pid in target_ids else 0.0 for pid in predicted_ids[:k]]
    ideal_relevances = sorted(
        [1.0] * len(target_ids) + [0.0] * (k - len(target_ids)), reverse=True
    )[:k]

    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(predicted_ids, target_ids, k):
    """Average Precision@k."""
    relevances = [1.0 if pid in target_ids else 0.0 for pid in predicted_ids[:k]]
    precisions = []
    for i, rel in enumerate(relevances, 1):
        if rel:
            precisions.append(sum(relevances[:i]) / i)
    return float(np.mean(precisions)) if precisions else 0.0


def recall_at_k(predicted_ids, target_ids, k):
    """Recall@k — fraction of relevant docs found in top-k."""
    if not target_ids:
        return 0.0
    found = sum(1 for pid in predicted_ids[:k] if pid in target_ids)
    return found / len(target_ids)


def precision_at_k(predicted_ids, target_ids, k):
    """Precision@k — fraction of top-k that is relevant."""
    if k == 0:
        return 0.0
    found = sum(1 for pid in predicted_ids[:k] if pid in target_ids)
    return found / k


def reciprocal_rank(predicted_ids, target_ids):
    """Reciprocal rank of first relevant doc."""
    for rank, pid in enumerate(predicted_ids, start=1):
        if pid in target_ids:
            return 1.0 / rank
    return 0.0


# Baseline: exact search per-query
def run_baseline(corpus_embeddings, doc_ids, query_embeddings, queries, qrels, run_id):
    print("\n[baseline] running exact cosine search per-query...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"baseline_{run_id}.csv")

    fieldnames = (
        ["query_id", "query_text", "time_ms"]
        + [f"ndcg@{k}" for k in K_VALUES]
        + [f"map@{k}" for k in K_VALUES]
        + [f"recall@{k}" for k in K_VALUES]
        + [f"precision@{k}" for k in K_VALUES]
        + ["mrr"]
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    top_k_max = max(K_VALUES)
    n_queries = len(queries)

    rows_buffer = []
    total_time_ms = 0.0
    metric_sums = {f"ndcg@{k}": 0.0 for k in K_VALUES}
    metric_sums.update({f"map@{k}": 0.0 for k in K_VALUES})
    metric_sums.update({f"recall@{k}": 0.0 for k in K_VALUES})
    metric_sums.update({f"precision@{k}": 0.0 for k in K_VALUES})
    mrr_sum = 0.0

    for i, q in enumerate(queries, start=1):
        query_emb = query_embeddings[i - 1]
        target_ids = set(qrels.get(q["query_id"], []))

        start = time.time()
        scores = corpus_embeddings @ query_emb
        top_indices = np.argsort(scores)[::-1][:top_k_max]
        elapsed_ms = (time.time() - start) * 1000

        predicted = [doc_ids[idx] for idx in top_indices]

        row = {
            "query_id": q["query_id"],
            "query_text": q["query_text"],
            "time_ms": elapsed_ms,
        }

        for k in K_VALUES:
            ndcg = ndcg_at_k(predicted, target_ids, k)
            map_k = average_precision_at_k(predicted, target_ids, k)
            rec = recall_at_k(predicted, target_ids, k)
            prec = precision_at_k(predicted, target_ids, k)

            row[f"ndcg@{k}"] = f"{ndcg:.4f}"
            row[f"map@{k}"] = f"{map_k:.4f}"
            row[f"recall@{k}"] = f"{rec:.4f}"
            row[f"precision@{k}"] = f"{prec:.4f}"

            metric_sums[f"ndcg@{k}"] += ndcg
            metric_sums[f"map@{k}"] += map_k
            metric_sums[f"recall@{k}"] += rec
            metric_sums[f"precision@{k}"] += prec

        rr = reciprocal_rank(predicted, target_ids)
        row["mrr"] = f"{rr:.4f}"
        mrr_sum += rr
        total_time_ms += elapsed_ms

        rows_buffer.append(row)

        is_flush_point = (i % 500 == 0) or (i == n_queries)
        if is_flush_point:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerows(rows_buffer)

            batch_avg_time = sum(float(r["time_ms"]) for r in rows_buffer) / len(rows_buffer)
            batch_avg_recall = sum(float(r["recall@1"]) for r in rows_buffer) / len(rows_buffer)
            print(
                f"[baseline] {i}/{n_queries} queries processed | "
                f"last batch avg time={batch_avg_time:.2f} ms, "
                f"avg recall@1={batch_avg_recall:.4f} | saved to {csv_path}"
            )
            rows_buffer = []

    summary = {
        "total_time_sec": total_time_ms / 1000,
        "avg_time_per_query_ms": total_time_ms / n_queries,
        "memory_embeddings_mb": corpus_embeddings.nbytes / (1024 ** 2),
    }
    for k in K_VALUES:
        summary[f"ndcg@{k}"] = metric_sums[f"ndcg@{k}"] / n_queries
        summary[f"map@{k}"] = metric_sums[f"map@{k}"] / n_queries
        summary[f"recall@{k}"] = metric_sums[f"recall@{k}"] / n_queries
        summary[f"precision@{k}"] = metric_sums[f"precision@{k}"] / n_queries
    summary["mrr"] = mrr_sum / n_queries

    print("[baseline] done:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")

    return csv_path, summary


# TurboVec: quantized search per-query
def run_turbovec_eval(index, doc_ids, query_embeddings, queries, qrels, bit_width, run_id):
    print(f"\n[turbovec bw={bit_width}] running quantized search per-query...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"turbovec_bw{bit_width}_{run_id}.csv")

    fieldnames = (
        ["query_id", "query_text", "time_ms"]
        + [f"ndcg@{k}" for k in K_VALUES]
        + [f"map@{k}" for k in K_VALUES]
        + [f"recall@{k}" for k in K_VALUES]
        + [f"precision@{k}" for k in K_VALUES]
        + ["mrr"]
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    top_k_max = max(K_VALUES)
    n_queries = len(queries)

    rows_buffer = []
    total_time_ms = 0.0
    metric_sums = {f"ndcg@{k}": 0.0 for k in K_VALUES}
    metric_sums.update({f"map@{k}": 0.0 for k in K_VALUES})
    metric_sums.update({f"recall@{k}": 0.0 for k in K_VALUES})
    metric_sums.update({f"precision@{k}": 0.0 for k in K_VALUES})
    mrr_sum = 0.0

    for i, q in enumerate(queries, start=1):
        query_emb = query_embeddings[i - 1]
        target_ids = set(qrels.get(q["query_id"], []))

        query_embedding = np.expand_dims(query_emb, axis=0).astype(np.float32)

        start = time.time()
        scores, indices = index.search(query_embedding, k=top_k_max)
        elapsed_ms = (time.time() - start) * 1000

        predicted = [doc_ids[idx] for idx in indices[0]]

        row = {
            "query_id": q["query_id"],
            "query_text": q["query_text"],
            "time_ms": elapsed_ms,
        }

        for k in K_VALUES:
            ndcg = ndcg_at_k(predicted, target_ids, k)
            map_k = average_precision_at_k(predicted, target_ids, k)
            rec = recall_at_k(predicted, target_ids, k)
            prec = precision_at_k(predicted, target_ids, k)

            row[f"ndcg@{k}"] = f"{ndcg:.4f}"
            row[f"map@{k}"] = f"{map_k:.4f}"
            row[f"recall@{k}"] = f"{rec:.4f}"
            row[f"precision@{k}"] = f"{prec:.4f}"

            metric_sums[f"ndcg@{k}"] += ndcg
            metric_sums[f"map@{k}"] += map_k
            metric_sums[f"recall@{k}"] += rec
            metric_sums[f"precision@{k}"] += prec

        rr = reciprocal_rank(predicted, target_ids)
        row["mrr"] = f"{rr:.4f}"
        mrr_sum += rr
        total_time_ms += elapsed_ms

        rows_buffer.append(row)

        is_flush_point = (i % 500 == 0) or (i == n_queries)
        if is_flush_point:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerows(rows_buffer)

            batch_avg_time = sum(float(r["time_ms"]) for r in rows_buffer) / len(rows_buffer)
            batch_avg_recall = sum(float(r["recall@1"]) for r in rows_buffer) / len(rows_buffer)
            print(
                f"[turbovec bw={bit_width}] {i}/{n_queries} queries processed | "
                f"last batch avg time={batch_avg_time:.2f} ms, "
                f"avg recall@1={batch_avg_recall:.4f} | saved to {csv_path}"
            )
            rows_buffer = []

    summary = {
        "total_time_sec": total_time_ms / 1000,
        "avg_time_per_query_ms": total_time_ms / n_queries,
    }
    for k in K_VALUES:
        summary[f"ndcg@{k}"] = metric_sums[f"ndcg@{k}"] / n_queries
        summary[f"map@{k}"] = metric_sums[f"map@{k}"] / n_queries
        summary[f"recall@{k}"] = metric_sums[f"recall@{k}"] / n_queries
        summary[f"precision@{k}"] = metric_sums[f"precision@{k}"] / n_queries
    summary["mrr"] = mrr_sum / n_queries

    print(f"[turbovec bw={bit_width}] done:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")

    return csv_path, summary


def main():
    parser = argparse.ArgumentParser(description="Semantic Search Benchmark for TurboVec (MS MARCO)")
    parser.add_argument("--limit", type=int, default=None, help="optional cap on number of queries (quick dry run)")
    parser.add_argument("--quick-test", action="store_true", help="run on tiny subset for smoke testing")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"=== run_id={run_id} dataset=ms_marco model={MODEL_NAME} ===")

    print("[load] reading corpus, queries, and qrels from disk...")
    corpus = load_json("corpus.json")
    queries = load_json("queries.json")
    qrels = load_json("qrels.json")

    if args.quick_test:
        corpus = corpus[:100]
        corpus_doc_ids = {c["doc_id"] for c in corpus}
        # Filter queries to only those with relevant docs in the reduced corpus
        filtered_queries = []
        for q in queries:
            rel_docs = [d for d in qrels.get(q["query_id"], []) if d in corpus_doc_ids]
            if rel_docs:
                filtered_queries.append(q)
        queries = filtered_queries[:10]
        # Update qrels to only include docs in reduced corpus
        qrels = {q["query_id"]: [d for d in qrels.get(q["query_id"], []) if d in corpus_doc_ids]
                 for q in queries}
        print(f"[quick-test] limited to {len(corpus)} passages, {len(queries)} queries")
    elif args.limit:
        queries = queries[:args.limit]

    print(f"[load] {len(corpus)} passages, {len(queries)} queries")

    corpus_texts = [c["text"] for c in corpus]
    doc_ids = [c["doc_id"] for c in corpus]

    print(f"[model] loading {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    print("[encode] embedding corpus passages...")
    mem_before = get_rss_mb()
    corpus_embeddings = model.encode(
        corpus_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    ).astype(np.float32)
    mem_after = get_rss_mb()
    print(
        f"[encode] done. shape={corpus_embeddings.shape}, "
        f"array size={corpus_embeddings.nbytes / (1024 ** 2):.2f} MB, "
        f"process RSS delta={mem_after - mem_before:.2f} MB"
    )

    print("[encode] embedding queries...")
    query_texts = [q["query_text"] for q in queries]
    query_embeddings = model.encode(
        query_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    ).astype(np.float32)

    baseline_csv_path, baseline_summary = run_baseline(
        corpus_embeddings, doc_ids, query_embeddings, queries, qrels, run_id
    )

    baseline_mb = corpus_embeddings.nbytes / (1024 ** 2)
    n_docs, dim = corpus_embeddings.shape
    turbovec_results = []

    for bw in BIT_WIDTHS:
        print(f"\n--- bit_width={bw} ---")
        print("[turbovec] building index...")
        mem_before_index = get_rss_mb()
        index = TurboQuantIndex(dim=dim, bit_width=bw)
        index.add(corpus_embeddings)
        mem_after_index = get_rss_mb()

        turbovec_rss_delta_mb = mem_after_index - mem_before_index
        theoretical_turbovec_mb = (n_docs * dim * bw / 8) / (1024 ** 2)

        print(
            f"[turbovec] index built. "
            f"process RSS delta={turbovec_rss_delta_mb:.2f} MB, "
            f"theoretical size={theoretical_turbovec_mb:.2f} MB, "
            f"vs baseline float32={baseline_mb:.2f} MB "
            f"(~{baseline_mb / theoretical_turbovec_mb:.1f}x smaller in theory)"
        )

        print("[turbovec] running per-query search")
        csv_path, turbovec_summary = run_turbovec_eval(
            index, doc_ids, query_embeddings, queries, qrels, bw, run_id
        )

        turbovec_summary["bit_width"] = bw
        turbovec_results.append(turbovec_summary)

        print(f"\nTurboVec bit_width={bw} results:")
        for key, value in turbovec_summary.items():
            if key == "bit_width":
                continue
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")

    print("\n=== SUMMARY ===")
    print("Baseline (exact cosine search, float32, per-query):")
    for key, value in baseline_summary.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")

    print("\nTurboVec results by bit_width:")
    for tv in turbovec_results:
        print(f"  bit_width={tv['bit_width']}:")
        for key, value in tv.items():
            if key == "bit_width":
                continue
            if isinstance(value, float):
                print(f"      {key}: {value:.4f}")
            else:
                print(f"      {key}: {value}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, f"summary_{run_id}.json")

    memory_entries = {
        "baseline_embeddings": baseline_mb,
    }
    for tv in turbovec_results:
        bw = tv["bit_width"]
        memory_entries[f"turbovec_theoretical_bw{bw}"] = (n_docs * dim * bw / 8) / (1024 ** 2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "dataset": "ms_marco",
                "model": MODEL_NAME,
                "n_passages": len(corpus),
                "n_queries": len(queries),
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
