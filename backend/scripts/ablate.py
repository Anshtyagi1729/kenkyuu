"""Feature ablation measured on the 40-query evaluation set.

    python -m scripts.ablate

`scripts.train_ranker` already ablates, but it scores on the SYNTHETIC validation
split - queries built by lifting a sentence out of an abstract. Text matching nearly
solves that task by construction (0.93 nDCG), which leaves no room for any other
signal to show value, so that ablation reports citation features as worth ~0.000.
That is a true statement about the synthetic task and a misleading one about the
product.

This script retrains each variant and scores it on the hand-written eval set, where
the conditional behaviour is actually exercised - some queries want a foundational
paper, others want a niche one, and a feature group can therefore be measured on
whether it helps the system tell those apart.

Two things make the comparison controlled:

  * every variant re-ranks the SAME cached 60-candidate pool, so differences come
    from the model and not from retrieval variance;
  * the control row is that pool in cross-encoder order, which is exactly what the
    system did before Phase 4.

Scores are reported split by landmark vs obscure queries, because the whole finding
of Phase 3 was that citation evidence helps one half and destroys the other. An
aggregate number hides the trade this project exists to measure.
"""

import argparse
import logging
import pickle
import time
from pathlib import Path

from app.config import BASE_DIR
from app.evaluation import harness
from app.ranking import rerank
from app.ranking.model import train
from app.storage import vector_store

logger = logging.getLogger("ablate")

EVAL_SET_PATH = Path(BASE_DIR) / "data" / "eval_set.json"
TRAINING_PATH = Path(BASE_DIR) / "data" / "training_rows.pkl"
POOL_CACHE = Path(BASE_DIR) / "data" / "eval_pool.pkl"
REPORT_PATH = Path(BASE_DIR) / "data" / "ablation_eval.md"

# Same grouping as scripts.train_ranker, and deliberately so: the two ablations are
# meant to be read side by side, and regrouping between them would make the synthetic
# and eval-set columns incomparable.
FEATURE_GROUPS = {
    "text relevance": ["rerank_score", "bm25_score", "title_overlap"],
    "citation evidence": ["log_citations", "log_cnci", "has_cnci", "influential_ratio"],
    "graph centrality": ["log_pagerank", "log_in_degree"],
    "paper properties": ["age_years", "has_venue"],
    "query shape": ["query_words", "is_recency_query", "is_comparison_query", "is_foundational_query"],
}

# Queries whose correct answer sits in the 55th-96th citation percentile rather than
# the 99.9th. They exist to catch popularity collapse, so they are scored separately.
OBSCURE_PREFIX = "obscure-"

# Leave-one-out says what each group is worth in the presence of all the others. It
# does NOT say which combination is best - if two groups are both mildly harmful,
# removing either alone still leaves the other doing damage. These are the
# combinations the leave-one-out table points at, tested directly so the chosen
# configuration is measured rather than inferred.
CANDIDATE_CONFIGS = {
    "text + graph centrality only": ["paper properties", "citation evidence", "query shape"],
    "text + all citation features": ["paper properties", "query shape"],
    "text relevance only": ["citation evidence", "graph centrality", "paper properties", "query shape"],
}


def build_pools(queries: list[harness.EvalQuery], pool_size: int) -> dict:
    """Retrieves and featurizes each query's candidates once, then caches.

    Cached because retrieval here means an embedding pass plus a cross-encoder over
    120 candidates per query. Paying that once instead of once per variant is the
    difference between a minute and ten, and - more importantly - it guarantees every
    variant is judged on byte-identical candidates.
    """
    pools = {}
    start = time.time()
    for i, query in enumerate(queries, 1):
        pool = rerank.candidate_pool(
            query.query, pool_size=pool_size, collection=vector_store.EVAL_COLLECTION_NAME
        )
        pools[query.id] = {
            # Only what scoring needs. The chunk text is the bulky part and nothing
            # downstream reads it, so it is dropped rather than pickled 40 times.
            "paper_ids": [h["paper_id"] for h in pool],
            "rows": rerank.feature_rows(query.query, pool),
        }
        if i % 10 == 0:
            logger.info("  pooled %d/%d queries (%.1fs)", i, len(queries), time.time() - start)
    return pools


def load_pools(queries: list[harness.EvalQuery], pool_size: int, rebuild: bool) -> dict:
    if POOL_CACHE.exists() and not rebuild:
        blob = pickle.loads(POOL_CACHE.read_bytes())
        # A cache built for a different eval set or pool depth would silently score
        # the wrong thing, so it is validated rather than trusted.
        if blob.get("pool_size") == pool_size and set(blob["pools"]) == {q.id for q in queries}:
            logger.info("Loaded cached pools for %d queries", len(blob["pools"]))
            return blob["pools"]
        logger.info("Cached pools do not match this eval set - rebuilding")

    logger.info("Building candidate pools (%d queries, depth %d)...", len(queries), pool_size)
    pools = build_pools(queries, pool_size)
    POOL_CACHE.write_bytes(pickle.dumps({"pool_size": pool_size, "pools": pools}))
    logger.info("Cached pools to %s", POOL_CACHE.name)
    return pools


def _dedupe(paper_ids: list[str]) -> list[str]:
    """Chunk ranking -> paper ranking, keeping each paper's best position."""
    seen, ranked = set(), []
    for paper_id in paper_ids:
        if paper_id not in seen:
            seen.add(paper_id)
            ranked.append(paper_id)
    return ranked


def control_system(pools: dict, by_query: dict) -> harness.RetrievalFn:
    """The pool in cross-encoder order - the system as it stood before Phase 4."""

    def run(query: str, k: int) -> list[str]:
        return _dedupe(pools[by_query[query]]["paper_ids"])[:k]

    return run


def model_system(ranker, pools: dict, by_query: dict) -> harness.RetrievalFn:
    """The same pool, reordered by a trained variant."""

    def run(query: str, k: int) -> list[str]:
        pool = pools[by_query[query]]
        scores = ranker.score(rerank.project(pool["rows"], ranker.feature_names))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return _dedupe([pool["paper_ids"][i] for i in order])[:k]

    return run


def split_scores(report: harness.EvalReport) -> dict[str, float]:
    """Overall, landmark-only and obscure-only nDCG from one report."""
    landmark = [s.ndcg for s in report.per_query if not s.query_id.startswith(OBSCURE_PREFIX)]
    obscure = [s.ndcg for s in report.per_query if s.query_id.startswith(OBSCURE_PREFIX)]
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    return {
        "ndcg": report.ndcg,
        "mrr": report.mrr,
        "landmark": mean(landmark),
        "obscure": mean(obscure),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--pool-size", type=int, default=rerank.POOL_SIZE)
    parser.add_argument("--rebuild-pools", action="store_true", help="ignore the cached candidate pools")
    parser.add_argument(
        "--seeds",
        type=int,
        default=0,
        help="retrain the shipped configuration this many times with different train/valid "
             "splits, to measure how much of a difference this eval set can actually resolve",
    )
    parser.add_argument(
        "--sweep-rounds",
        nargs="*",
        type=int,
        default=None,
        help="train the shipped configuration for these fixed round counts, early stopping OFF, "
             "and score each on the eval set",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not TRAINING_PATH.exists():
        raise SystemExit(f"No training data at {TRAINING_PATH} - run scripts.build_training_set first")

    queries = harness.load_eval_set(EVAL_SET_PATH)
    by_query = {q.query: q.id for q in queries}
    n_obscure = sum(q.id.startswith(OBSCURE_PREFIX) for q in queries)
    logger.info("%d eval queries (%d landmark / %d obscure)\n", len(queries), len(queries) - n_obscure, n_obscure)

    pools = load_pools(queries, args.pool_size, args.rebuild_pools)

    blob = pickle.loads(TRAINING_PATH.read_bytes())
    groups, names = blob["rows"], blob["feature_names"]
    logger.info("Training on %d query groups, %d features\n", len(groups), len(names))

    rows: list[tuple[str, dict, float | None]] = []
    seed_stats: dict | None = None

    control = split_scores(harness.evaluate(control_system(pools, by_query), queries))
    rows.append(("cross-encoder only (control)", control, None))

    full_ranker, full_info = train(groups, names, num_rounds=args.rounds)
    full = split_scores(harness.evaluate(model_system(full_ranker, pools, by_query), queries))
    rows.append(("learned ranker (all features)", full, full["ndcg"] - control["ndcg"]))

    for label, feats in FEATURE_GROUPS.items():
        variant, _ = train(groups, names, num_rounds=args.rounds, exclude=feats)
        scores = split_scores(harness.evaluate(model_system(variant, pools, by_query), queries))
        rows.append((f"  without {label}", scores, scores["ndcg"] - full["ndcg"]))

    header = f"{'system':<32}{'nDCG@10':>10}{'MRR':>9}{'landmark':>11}{'obscure':>10}{'delta':>10}"
    print("\n" + header)
    print("-" * len(header))
    for label, s, delta in rows:
        d = f"{delta:+.4f}" if delta is not None else "--"
        print(f"{label:<32}{s['ndcg']:>10.4f}{s['mrr']:>9.4f}{s['landmark']:>11.4f}{s['obscure']:>10.4f}{d:>10}")

    ablated = rows[2:]
    if ablated:
        worst = min(ablated, key=lambda r: r[2])
        print(f"\nMost valuable group on the eval set: {worst[0].strip()[8:]!r} ({worst[2]:+.4f})")

    print("\nDIRECT CONFIGURATIONS — the combinations the table above points at")
    print(header)
    print("-" * len(header))
    config_rows = []
    for label, dropped in CANDIDATE_CONFIGS.items():
        excluded = [f for group in dropped for f in FEATURE_GROUPS[group]]
        variant, _ = train(groups, names, num_rounds=args.rounds, exclude=excluded)
        s = split_scores(harness.evaluate(model_system(variant, pools, by_query), queries))
        delta = s["ndcg"] - full["ndcg"]
        config_rows.append((label, s, delta))
        print(f"{label:<32}{s['ndcg']:>10.4f}{s['mrr']:>9.4f}{s['landmark']:>11.4f}{s['obscure']:>10.4f}{delta:>+10.4f}")

    best = max([("learned ranker (all features)", full, 0.0), *config_rows], key=lambda r: r[1]["ndcg"])
    print(f"\nBest measured configuration: {best[0]!r} at {best[1]['ndcg']:.4f} nDCG@10")
    rows = rows + [("", {"ndcg": 0, "mrr": 0, "landmark": 0, "obscure": 0}, None)] + config_rows

    if args.sweep_rounds:
        shipped = [f for g, feats in FEATURE_GROUPS.items()
                   if g not in ("text relevance", "graph centrality") for f in feats]
        print("\nROUND SWEEP — shipped configuration, early stopping disabled")
        print(header)
        print("-" * len(header))
        sweep_rows = []
        for n in args.sweep_rounds:
            variant, _ = train(groups, names, num_rounds=n, exclude=shipped, early_stopping=False)
            s = split_scores(harness.evaluate(model_system(variant, pools, by_query), queries))
            sweep_rows.append((f"text+centrality, {n} rounds", s, s["ndcg"] - full["ndcg"]))
            print(f"{sweep_rows[-1][0]:<32}{s['ndcg']:>10.4f}{s['mrr']:>9.4f}"
                  f"{s['landmark']:>11.4f}{s['obscure']:>10.4f}{sweep_rows[-1][2]:>+10.4f}")
        rows = rows + [("", {"ndcg": 0, "mrr": 0, "landmark": 0, "obscure": 0}, None)] + sweep_rows

    if args.seeds:
        shipped = [f for g, feats in FEATURE_GROUPS.items()
                   if g not in ("text relevance", "graph centrality") for f in feats]
        print(f"\nSEED VARIANCE — shipped configuration retrained {args.seeds}x")
        print("  Same data, same features, only the train/valid split differs. The spread")
        print("  here is the floor on what a difference between two rows above can mean.")
        scores = []
        for seed in range(args.seeds):
            variant, _ = train(groups, names, num_rounds=args.rounds, exclude=shipped, seed=seed)
            s = split_scores(harness.evaluate(model_system(variant, pools, by_query), queries))
            scores.append(s["ndcg"])
            print(f"    seed {seed}: nDCG@10 = {s['ndcg']:.4f}   landmark {s['landmark']:.4f}   obscure {s['obscure']:.4f}")
        mean = sum(scores) / len(scores)
        spread = max(scores) - min(scores)
        sd = (sum((x - mean) ** 2 for x in scores) / len(scores)) ** 0.5
        seed_stats = {"n": len(scores), "mean": mean, "sd": sd, "spread": spread,
                      "lo": min(scores), "hi": max(scores)}
        print(f"\n    mean {mean:.4f}   sd {sd:.4f}   range {spread:.4f} "
              f"({min(scores):.4f} to {max(scores):.4f})")
        print(f"    => differences smaller than about {spread:.3f} nDCG are not resolvable here.")

    _write_report(rows, full_info, len(queries), n_obscure, args.pool_size, seed_stats)
    print(f"Wrote {REPORT_PATH.name}")


def _write_report(rows, full_info: dict, n_queries: int, n_obscure: int, pool_size: int,
                  seed_stats: dict | None = None) -> None:
    """Emits the table as markdown, ready to paste into data/results.md."""
    lines = [
        "## Ablation on the evaluation set",
        "",
        f"Every variant retrained from scratch and scored on the same {n_queries}-query set "
        f"({n_queries - n_obscure} landmark / {n_obscure} obscure), re-ranking an identical cached "
        f"{pool_size}-candidate pool. Control is that pool in cross-encoder order.",
        "",
        "| System | nDCG@10 | MRR | landmark | obscure | Δ |",
        "|---|---|---|---|---|---|",
    ]
    for label, s, delta in rows:
        if not label.strip():
            lines.append("| | | | | | |")
            continue
        d = f"{delta:+.4f}" if delta is not None else "—"
        lines.append(
            f"| {label.strip()} | {s['ndcg']:.4f} | {s['mrr']:.4f} | "
            f"{s['landmark']:.4f} | {s['obscure']:.4f} | {d} |"
        )
    lines += [
        "",
        f"Full model trained on {full_info['train_groups']} groups, validated on "
        f"{full_info['valid_groups']} (best iteration {full_info['best_iteration']}).",
        "",
    ]

    # The noise floor travels WITH the table, always. This file exists to be pasted
    # into a report, and the per-group deltas above are smaller than the run-to-run
    # variance - so a reader who sees the table without this paragraph will conclude
    # the opposite of what the data supports.
    if seed_stats:
        lines += [
            "### Noise floor — read before interpreting any Δ above",
            "",
            f"Retraining the SAME configuration {seed_stats['n']} times, changing only the "
            f"train/validation split, gives nDCG@10 from {seed_stats['lo']:.4f} to "
            f"{seed_stats['hi']:.4f} — mean {seed_stats['mean']:.4f}, sd {seed_stats['sd']:.4f}, "
            f"range **{seed_stats['spread']:.4f}**.",
            "",
            f"Any Δ in the table smaller than about {seed_stats['spread']:.3f} nDCG is not "
            "distinguishable from which random split was drawn, and must not be reported as a "
            "feature's contribution. On this run that covers every per-group Δ except "
            "*text relevance*.",
            "",
        ]
    else:
        lines += [
            "### No noise floor was measured on this run",
            "",
            "Re-run with `--seeds 8` before quoting any Δ above. Run-to-run variance from the "
            "train/validation split alone has measured larger than every per-group difference "
            "here except *text relevance*.",
            "",
        ]
    REPORT_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
