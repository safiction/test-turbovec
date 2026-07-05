"""
Scaled-up version of toy.py: benchmarks exact (baseline) search vs TurboVec
on the full validation ("small") or train ("large") RAG splits.
Runs baseline per-query (fair comparison) and tests TurboVec at bit-widths 2 and 4.

Usage:
    py rag_search.py --mode small
    py rag_search.py --mode large --dim 512
    py rag_search.py --mode small --limit 100        # quick run
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
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "rag_search"
RESULTS_DIR = "results/rag_search"

context_paths = {
    "small": str(DATA_DIR / "validation_contexts.json"),
    "large": str(DATA_DIR / "train_contexts.json"),
}

questions_paths = {
    "small": str(DATA_DIR / "validation_questions.json"),
    "large": str(DATA_DIR / "train_questions.json"),
}


def load_data(path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_rss_mb() -> float:
    """Current resident memory (RSS) of this process, in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def exact_search(query, vectors, ids, k):
    if query.ndim == 2:
        query = query[0]
    scores = vectors @ query
    top_indices = np.argsort(scores)[::-1][:k]
    return [ids[i] for i in top_indices]


def recall_at_k(predicted, target, k):
    return int(target in predicted[:k])


def reciprocal_rank(predicted, target):
    for rank, item in enumerate(predicted, start=1):
        if item == target:
            return 1 / rank
    return 0

# Baseline: run PER-QUESTION to match TurboVec latency measurement.
def run_baseline(context_embeddings, context_ids, query_embeddings, questions, k_values, mode, dim, run_id, save_every=20):
    print("\n[baseline] running exact search per-question (fair comparison)...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"baseline_{mode}_dim{dim}_{run_id}.csv")

    fieldnames = (
        ["question_id", "question_text", "target_context_id", "time_ms"]
        + [f"recall@{k}" for k in k_values]
        + ["mrr"]
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    top_k_max = max(k_values)
    n_questions = len(questions)

    rows_buffer = []
    total_time_ms = 0.0
    recall_sums = {k: 0 for k in k_values}
    mrr_sum = 0.0

    for i, q in enumerate(questions, start=1):
        query = query_embeddings[i - 1]
        if query.ndim == 2:
            query = query[0]

        start = time.time()
        scores = context_embeddings @ query
        top_indices = np.argsort(scores)[::-1][:top_k_max]
        elapsed_ms = (time.time() - start) * 1000

        predicted = [context_ids[j] for j in top_indices]
        target = q["context_id"]

        row = {
            "question_id": q.get("question_id", i),
            "question_text": q["question_text"],
            "target_context_id": target,
            "time_ms": elapsed_ms,
        }
        for k in k_values:
            r = recall_at_k(predicted, target, k)
            row[f"recall@{k}"] = r
            recall_sums[k] += r
        rr = reciprocal_rank(predicted, target)
        row["mrr"] = rr
        mrr_sum += rr
        total_time_ms += elapsed_ms

        rows_buffer.append(row)

        is_flush_point = (i % save_every == 0) or (i == n_questions)
        if is_flush_point:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerows(rows_buffer)

            batch_avg_time = sum(r["time_ms"] for r in rows_buffer) / len(rows_buffer)
            batch_avg_recall1 = sum(r.get(f"recall@{k_values[0]}", 0) for r in rows_buffer) / len(rows_buffer)
            print(
                f"[baseline] {i}/{n_questions} questions processed | "
                f"last batch avg time={batch_avg_time:.2f} ms, "
                f"avg recall@{k_values[0]}={batch_avg_recall1:.2f} | saved to {csv_path}"
            )
            rows_buffer = []

    summary = {
        "total_time_sec": total_time_ms / 1000,
        "avg_time_per_query_ms": total_time_ms / n_questions,
        "memory_embeddings_mb": context_embeddings.nbytes / (1024 ** 2),
    }
    for k in k_values:
        summary[f"recall@{k}"] = recall_sums[k] / n_questions
    summary["mrr"] = mrr_sum / n_questions

    print("[baseline] done:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")

    return csv_path, summary


# TurboVec: run per-question, save to CSV every `save_every` questions.
def run_turbovec_eval(index, context_ids, model, questions, k_values, mode, dim, bit_width, run_id, save_every=20):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"turbovec_{mode}_dim{dim}_bw{bit_width}_{run_id}.csv")

    fieldnames = (
        ["question_id", "question_text", "target_context_id", "time_ms"]
        + [f"recall@{k}" for k in k_values]
        + ["mrr"]
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    top_k_max = max(k_values)
    n_questions = len(questions)

    rows_buffer = []
    # running sums for the final aggregated summary
    total_time_ms = 0.0
    recall_sums = {k: 0 for k in k_values}
    mrr_sum = 0.0

    for i, q in enumerate(questions, start=1):
        query_embedding = model.encode(q["question_text"], normalize_embeddings=True)
        query_embedding = np.expand_dims(query_embedding, axis=0).astype(np.float32)

        start = time.time()
        scores, indices = index.search(query_embedding, k=top_k_max)
        elapsed_ms = (time.time() - start) * 1000

        predicted = [context_ids[j] for j in indices[0]]
        target = q["context_id"]

        row = {
            "question_id": q.get("question_id", i),
            "question_text": q["question_text"],
            "target_context_id": target,
            "time_ms": elapsed_ms,
        }
        for k in k_values:
            r = recall_at_k(predicted, target, k)
            row[f"recall@{k}"] = r
            recall_sums[k] += r
        rr = reciprocal_rank(predicted, target)
        row["mrr"] = rr
        mrr_sum += rr
        total_time_ms += elapsed_ms

        rows_buffer.append(row)

        is_flush_point = (i % save_every == 0) or (i == n_questions)
        if is_flush_point:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerows(rows_buffer)

            batch_avg_time = sum(r["time_ms"] for r in rows_buffer) / len(rows_buffer)
            batch_avg_recall1 = sum(r.get(f"recall@{k_values[0]}", 0) for r in rows_buffer) / len(rows_buffer)
            print(
                f"[turbovec] {i}/{n_questions} questions processed | "
                f"last batch avg time={batch_avg_time:.2f} ms, "
                f"avg recall@{k_values[0]}={batch_avg_recall1:.2f} | saved to {csv_path}"
            )
            rows_buffer = []

    summary = {
        "total_time_sec": total_time_ms / 1000,
        "avg_time_per_query_ms": total_time_ms / n_questions,
    }
    for k in k_values:
        summary[f"recall@{k}"] = recall_sums[k] / n_questions
    summary["mrr"] = mrr_sum / n_questions

    return csv_path, summary


def main():
    parser = argparse.ArgumentParser(description="Scale toy RAG + TurboVec benchmark to full dataset.")
    parser.add_argument("--mode", choices=["small", "large"], required=True,
                         help="small = validation split (3971 ctx / 2000 q), large = train split (9078 ctx / 5000 q)")
    parser.add_argument("--dim", type=int, default=384, help="embedding dimension (truncate_dim)")
    parser.add_argument("--k", type=str, default="1,5,10", help="comma-separated k values for Recall@k")
    parser.add_argument("--save-every", type=int, default=20, help="flush TurboVec results to CSV every N questions")
    parser.add_argument("--limit", type=int, default=None, help="optional cap on number of questions (quick dry run)")
    parser.add_argument("--quick-test", action="store_true", help="run on 10 contexts and 1 matching question for a smoke test")
    args = parser.parse_args()

    k_values = [int(x) for x in args.k.split(",")]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"=== run_id={run_id} mode={args.mode} dim={args.dim} k={k_values} ===")

    print("[load] reading contexts and questions from disk...")
    contexts = load_data(context_paths[args.mode])
    questions = load_data(questions_paths[args.mode])

    if args.quick_test:
        contexts = contexts[:10]
        context_ids_set = {c["context_id"] for c in contexts}
        matching_questions = [q for q in questions if q["context_id"] in context_ids_set]
        if not matching_questions:
            print("[quick-test] ERROR: no question matches the first 10 contexts. aborting.")
            return
        questions = matching_questions[:1]
        print(f"[quick-test] limited to {len(contexts)} contexts, {len(questions)} question")
    elif args.limit:
        questions = questions[: args.limit]

    print(f"[load] {len(contexts)} contexts, {len(questions)} questions")

    context_texts = [c["text"] for c in contexts]
    context_ids = [c["context_id"] for c in contexts]

    print("[model] loading nomic-ai/nomic-embed-text-v1.5 ...")
    model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5", truncate_dim=args.dim)

    print("[encode] embedding contexts...")
    mem_before_ctx = get_rss_mb()
    context_embeddings = model.encode(
        context_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    ).astype(np.float32)
    mem_after_ctx = get_rss_mb()
    print(
        f"[encode] done. shape={context_embeddings.shape}, "
        f"array size={context_embeddings.nbytes / (1024 ** 2):.2f} MB, "
        f"process RSS delta={mem_after_ctx - mem_before_ctx:.2f} MB"
    )

    print("[encode] embedding questions (needed for the one-off baseline pass)...")
    question_texts = [q["question_text"] for q in questions]
    query_embeddings = model.encode(
        question_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    ).astype(np.float32)

    baseline_csv_path, baseline_summary = run_baseline(
        context_embeddings, context_ids, query_embeddings, questions, k_values,
        args.mode, args.dim, run_id, args.save_every,
    )

    baseline_mb = context_embeddings.nbytes / (1024 ** 2)
    turbovec_results = []

    BIT_WIDTHS = [2, 4]
    for bw in BIT_WIDTHS:
        print(f"\n[turbovec] building index for bit_width={bw}...")
        mem_before_index = get_rss_mb()
        index = TurboQuantIndex(dim=context_embeddings.shape[1], bit_width=bw)
        index.add(context_embeddings)
        mem_after_index = get_rss_mb()

        turbovec_rss_delta_mb = mem_after_index - mem_before_index
        n_ctx, dim = context_embeddings.shape
        theoretical_turbovec_mb = (n_ctx * dim * bw / 8) / (1024 ** 2)

        print(
            f"[turbovec] index built. "
            f"process RSS delta={turbovec_rss_delta_mb:.2f} MB, "
            f"theoretical size={theoretical_turbovec_mb:.2f} MB, "
            f"vs baseline float32={baseline_mb:.2f} MB "
            f"(~{baseline_mb / theoretical_turbovec_mb:.1f}x smaller in theory)"
        )

        print(f"\n[turbovec] running per-question search for bit_width={bw}, flushing every {args.save_every} questions to CSV...")
        csv_path, turbovec_summary = run_turbovec_eval(
            index, context_ids, model, questions, k_values,
            args.mode, args.dim, bw, run_id, args.save_every,
        )

        turbovec_summary["bit_width"] = bw
        turbovec_results.append(turbovec_summary)

        print(f"\nTurboVec bit_width={bw} results:")
        for key, value in turbovec_summary.items():
            print(f"    {key}: {value}")

    print("\n=== SUMMARY ===")
    print("Baseline (exact search, float32, per-query):")
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
    summary_path = os.path.join(RESULTS_DIR, f"summary_{args.mode}_dim{args.dim}_{run_id}.json")

    memory_entries = {
        "baseline_embeddings": baseline_mb,
    }
    for tv in turbovec_results:
        bw = tv["bit_width"]
        n_ctx, dim = context_embeddings.shape
        memory_entries[f"turbovec_theoretical_bw{bw}"] = (n_ctx * dim * bw / 8) / (1024 ** 2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "mode": args.mode,
                "dim": args.dim,
                "n_contexts": len(contexts),
                "n_questions": len(questions),
                "baseline": baseline_summary,
                "turbovec": turbovec_results,
                "memory_mb": memory_entries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nPer-question baseline results: {baseline_csv_path}")
    print(f"Run summary: {summary_path}")


if __name__ == "__main__":
    main()