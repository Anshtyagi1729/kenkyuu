"""Trains the ranker and runs the feature ablation.

    python -m scripts.train_ranker

Two outputs:
  1. A trained model saved to data/ranker.pkl
  2. The ablation table - retrain with each feature group removed, and report how
     much validation nDCG each group was worth.

The ablation matters more than the model. Feature importance only describes the model
that happened to get built; removing a feature and retraining measures what it was
actually worth. Those disagree whenever features are correlated, which citation
features very much are.
"""

import argparse
import logging
import pickle
from pathlib import Path

from app.config import BASE_DIR
from app.ranking.model import TrainedRanker, train

logger = logging.getLogger("train_ranker")

TRAINING_PATH = Path(BASE_DIR) / "data" / "training_rows.pkl"
MODEL_PATH = Path(BASE_DIR) / "data" / "ranker.pkl"

# Grouped so the ablation answers questions worth asking. Dropping one correlated
# feature proves little when three others carry the same signal; dropping the whole
# group answers "was citation evidence worth anything at all".
FEATURE_GROUPS = {
    "text relevance": ["rerank_score", "bm25_score", "title_overlap"],
    "citation evidence": ["log_citations", "log_cnci", "has_cnci", "influential_ratio"],
    "graph centrality": ["log_pagerank", "log_in_degree"],
    "paper properties": ["age_years", "has_venue"],
    "query shape": ["query_words", "is_recency_query", "is_comparison_query", "is_foundational_query"],
}

# The groups the SHIPPED model is trained without - deliberately none.
#
# An earlier version of this file excluded the citation-volume features, because
# `scripts.ablate` scored text + graph centrality at 0.8466 nDCG@10 against 0.8389
# for the full feature set. That exclusion has been reverted, and the reason is worth
# recording rather than quietly deleting.
#
# Retraining the SAME configuration under eight different train/validation splits
# produces eval-set scores from 0.7508 to 0.8586 - a range of 0.108 nDCG, standard
# deviation 0.043. Every difference between feature-group configurations measured so
# far (+0.0304 for dropping citation evidence, -0.0291 for dropping centrality,
# +0.0078 for the configuration above) is several times smaller than that spread.
# They are not measurements of anything; they are which random split got drawn.
#
# Choosing a feature set by those differences would be selecting a model on the test
# set using a signal indistinguishable from noise, and would then report the
# resulting number as though the test set were still held out. Training on
# everything makes no selection at all, which is the only honest option until the
# eval set is large enough to resolve a difference this size.
#
# What IS resolvable at this sample size: the learned ranker beats the cross-encoder
# by +0.1365 nDCG averaged over seeds (outside the 0.108 range), text relevance is
# worth -0.6255 when removed, and the obscure queries hold at 1.0000 in all eight
# seeds and every configuration - so the system does not collapse onto popularity.
DEFAULT_EXCLUDED_GROUPS: tuple[str, ...] = ()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument(
        "--groups",
        nargs="*",
        choices=list(FEATURE_GROUPS),
        default=None,
        help="feature groups to TRAIN ON (default: all of them - see DEFAULT_EXCLUDED_GROUPS)",
    )
    parser.add_argument(
        "--all-features",
        action="store_true",
        help="train on all 15 features, ignoring the measured-best configuration",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not TRAINING_PATH.exists():
        raise SystemExit(f"No training data at {TRAINING_PATH} - run scripts.build_training_set first")

    blob = pickle.loads(TRAINING_PATH.read_bytes())
    groups, names = blob["rows"], blob["feature_names"]
    rows = sum(len(g["group"]) for g in groups)
    print(f"{len(groups)} query groups, {rows} candidate rows, {len(names)} features\n")

    if args.all_features:
        shipped_groups = list(FEATURE_GROUPS)
    elif args.groups:
        shipped_groups = args.groups
    else:
        shipped_groups = [g for g in FEATURE_GROUPS if g not in DEFAULT_EXCLUDED_GROUPS]

    dropped = [f for g, feats in FEATURE_GROUPS.items() if g not in shipped_groups for f in feats]
    print(f"Training on groups: {', '.join(shipped_groups)}")
    if dropped:
        print(f"Excluded {len(dropped)} features (see DEFAULT_EXCLUDED_GROUPS for why)\n")

    ranker, info = train(groups, names, num_rounds=args.rounds, exclude=dropped or None)
    baseline = info["valid_ndcg"]
    print(f"SHIPPED MODEL   valid nDCG@10 = {baseline:.4f}"
          f"   (best iter {info['best_iteration']}, "
          f"{info['train_groups']} train / {info['valid_groups']} valid groups)")
    print("  NOTE: this validation score is on SYNTHETIC queries and is not a product\n"
          "  metric - text alone nearly solves that task. See scripts.ablate for the\n"
          "  eval-set numbers that are.\n")

    print("FEATURE IMPORTANCE (gain %) — describes the model that got built")
    for name, pct in ranker.importance()[:10]:
        if pct > 0.1:
            print(f"  {name:<24}{pct:>6.1f}%")

    print("\nABLATION (synthetic split) — retrain without each group")
    print("  Read scripts.ablate instead for the eval-set version; on the synthetic")
    print("  task text dominates and every other group measures near zero.")
    print(f"  {'removed group':<22}{'valid nDCG':>12}{'delta':>10}")
    print("  " + "-" * 44)
    results = []
    for label, feats in FEATURE_GROUPS.items():
        if label not in shipped_groups:
            continue
        _, ablated = train(groups, names, num_rounds=args.rounds, exclude=feats)
        score = ablated["valid_ndcg"]
        if score is None:
            continue
        delta = score - baseline
        results.append((label, score, delta))
        print(f"  {label:<22}{score:>12.4f}{delta:>+10.4f}")

    if results:
        worst = min(results, key=lambda r: r[2])
        print(f"\n  Most valuable group: {worst[0]} (removing it costs {worst[2]:+.4f})")

    ranker.save(MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
