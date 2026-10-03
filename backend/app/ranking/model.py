"""LambdaMART ranker over text + scientometric features.

Why a learned model rather than a tuned weight: the evaluation established that a
fixed citation weight TRADES rather than improves - every weight that helped queries
seeking foundational papers hurt queries seeking specific niche work, monotonically
in both directions. No single point on that trade-off is acceptable.

A gradient-boosted tree can express what a weight cannot: conditional rules. It can
learn "when the query looks foundational AND centrality is high, raise this" while
also learning "when the query closely matches a title, trust the text score and
ignore citations entirely". Those are different rules for different queries, which is
exactly the structure the measurements demanded.

LambdaMART (Burges, 2010) optimizes ranking quality directly rather than per-item
accuracy - it cares where the right answer lands in the list, not whether each
candidate is individually classified correctly. LightGBM's `lambdarank` objective is
that algorithm; it trains on CPU in seconds at this data size.
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

logger = logging.getLogger(__name__)

# Deliberately small trees and few leaves. With ~1,000 query groups and 15 features,
# a deeper forest memorizes individual papers instead of learning a ranking policy -
# and memorizing "paper X is good" is precisely the popularity shortcut in another
# form.
DEFAULT_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [10],
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
}


@dataclass
class TrainedRanker:
    booster: lgb.Booster
    feature_names: list[str]

    def score(self, feature_rows: list[list[float]]) -> list[float]:
        if not feature_rows:
            return []
        return self.booster.predict(np.array(feature_rows, dtype=float)).tolist()

    def importance(self) -> list[tuple[str, float]]:
        """Feature importance by total gain - how much each feature improved the
        objective across all splits. Gain rather than split count: a feature used
        once at the root can matter far more than one used often near the leaves."""
        gains = self.booster.feature_importance(importance_type="gain")
        total = float(gains.sum()) or 1.0
        return sorted(
            ((name, 100.0 * g / total) for name, g in zip(self.feature_names, gains)),
            key=lambda kv: kv[1],
            reverse=True,
        )

    def save(self, path: Path) -> None:
        path.write_bytes(pickle.dumps({"model": self.booster.model_to_string(), "features": self.feature_names}))

    @staticmethod
    def load(path: Path) -> "TrainedRanker":
        blob = pickle.loads(path.read_bytes())
        return TrainedRanker(lgb.Booster(model_str=blob["model"]), blob["features"])


def _to_arrays(groups: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flattens query groups into LightGBM's (features, labels, group sizes) form.

    The group array is what makes this ranking rather than classification: it tells
    LightGBM which rows compete against each other, so the loss is computed within a
    query rather than across the whole dataset.
    """
    features, labels, sizes = [], [], []
    for group in groups:
        for row, label in group["group"]:
            features.append(row)
            labels.append(label)
        sizes.append(len(group["group"]))
    return np.array(features, dtype=float), np.array(labels, dtype=int), np.array(sizes, dtype=int)


def train(
    groups: list[dict],
    feature_names: list[str],
    num_rounds: int = 300,
    validation_fraction: float = 0.2,
    seed: int = 0,
    params: dict | None = None,
    exclude: list[str] | None = None,
    early_stopping: bool = True,
) -> tuple[TrainedRanker, dict]:
    """Trains the ranker. `exclude` drops named features - that is how the ablation
    measures each feature's contribution rather than merely reporting its importance,
    which only describes the model that was built and not what it was worth.

    `early_stopping=False` trains the full `num_rounds`. It exists because the
    validation split is SYNTHETIC - queries lifted from abstracts, which the text
    features alone score 0.93 on from the first boosting round. Early stopping
    therefore halts at iteration 1 or 2, before the model has learned much of
    anything about citation structure, on the grounds that a task which was never
    going to improve did not improve. Turning it off lets training continue past the
    point the wrong yardstick calls converged; whether that actually helps is
    measured on the eval set by `scripts.ablate --sweep-rounds`, not assumed.
    """
    keep_idx = [i for i, name in enumerate(feature_names) if not exclude or name not in exclude]
    kept_names = [feature_names[i] for i in keep_idx]

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    split = int(len(groups) * (1 - validation_fraction))
    train_groups = [groups[i] for i in order[:split]]
    valid_groups = [groups[i] for i in order[split:]]

    x_train, y_train, g_train = _to_arrays(train_groups)
    x_valid, y_valid, g_valid = _to_arrays(valid_groups)
    x_train, x_valid = x_train[:, keep_idx], x_valid[:, keep_idx]

    train_set = lgb.Dataset(x_train, label=y_train, group=g_train, feature_name=kept_names)
    valid_set = lgb.Dataset(x_valid, label=y_valid, group=g_valid, feature_name=kept_names, reference=train_set)

    evals: dict = {}
    booster = lgb.train(
        {**DEFAULT_PARAMS, **(params or {})},
        train_set,
        num_boost_round=num_rounds,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[
            *([lgb.early_stopping(30, verbose=False)] if early_stopping else []),
            lgb.record_evaluation(evals),
            lgb.log_evaluation(0),
        ],
    )

    history = evals.get("valid", {}).get("ndcg@10", [])
    # Without early stopping `best_iteration` is 0 (LightGBM only sets it when it
    # stops early), so fall back to the last round actually trained - otherwise the
    # reported valid_ndcg would index off the front of the history.
    best = booster.best_iteration or booster.num_trees() or len(history)
    return TrainedRanker(booster, kept_names), {
        "train_groups": len(train_groups),
        "valid_groups": len(valid_groups),
        "best_iteration": best,
        "valid_ndcg": history[min(best, len(history)) - 1] if history else None,
    }
