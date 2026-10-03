"""Field-normalized citation impact.

A raw citation count says almost nothing on its own, because it is dominated by age.
In this corpus a 2017 paper averages 324 citations and a 2025 paper averages 4 - an
80x gap that reflects time elapsed, not quality. Ranking by raw counts would just
rank by age.

Normalization fixes that by asking a different question: how does this paper compare
to OTHER PAPERS OF ITS OWN KIND? Same field, same year. A ratio of 1.0 is exactly
typical for its cohort; 40.0 means forty times its cohort's typical paper. Those
ratios are comparable across years in a way raw counts never are.

This is the standard instrument in the field - Web of Science calls it CNCI (Category
Normalized Citation Impact), Scopus calls it FWCI (Field-Weighted Citation Impact),
and the methodology follows Waltman et al. (2011), Scientometrics 87(3).

It is also the reason the corpus is scoped to deep learning rather than all of arXiv:
the cohort has to be a real peer group. "Average citations for a 2023 cs.LG paper" is
a meaningful number computed from ~800 papers; "average citations for a 2023 paper"
spans fields whose citation cultures differ by an order of magnitude and normalizes
against nothing at all.
"""

import logging
from dataclasses import dataclass

from app.storage import db

logger = logging.getLogger(__name__)

# Below this many papers a cohort's median is too noisy to normalize against, and a
# wrong denominator is worse than none - it would silently scale every paper in that
# cohort by a meaningless number.
MIN_COHORT_SIZE = 30


@dataclass
class Cohort:
    category: str
    year: int
    size: int
    median_citations: float

    @property
    def is_reliable(self) -> bool:
        return self.size >= MIN_COHORT_SIZE


def compute_cohorts() -> list[Cohort]:
    """Median citation count for every (category, year) group.

    MEDIAN, not mean, and this is not a detail. Citation distributions are extremely
    right-skewed: this corpus holds papers with 200,000+ citations alongside thousands
    with fewer than ten. A handful of famous papers drags the mean far above anything
    typical, so normalizing against it would make almost every paper look
    below-average. The median describes the actual middle of the field.
    """
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT primary_category, cohort_year, citation_count
            FROM paper_metrics
            WHERE citation_count IS NOT NULL
              AND primary_category IS NOT NULL
              AND cohort_year IS NOT NULL
            ORDER BY primary_category, cohort_year, citation_count
            """
        ).fetchall()

    grouped: dict[tuple[str, int], list[int]] = {}
    for row in rows:
        grouped.setdefault((row["primary_category"], row["cohort_year"]), []).append(row["citation_count"])

    return [
        Cohort(category, year, len(counts), _median(counts))
        for (category, year), counts in sorted(grouped.items())
    ]


def _median(sorted_values: list[int]) -> float:
    """Median of an already-sorted list (the SQL ORDER BY does the sorting)."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    mid = n // 2
    return float(sorted_values[mid]) if n % 2 else (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


POOLED_CATEGORY = "*"  # pseudo-category for the all-fields cohort of a given year


def compute_pooled_cohorts() -> list[Cohort]:
    """Per-year cohorts pooled across every category.

    A fallback peer group for papers whose own category has no usable cohort - most
    importantly the seeded landmarks, which carry no real arXiv category yet. Pooling
    across fields is a weaker comparison than like-for-like, since citation cultures
    differ between subfields, but it is anchored in the same year and so still removes
    the age effect that dominates raw counts. Far better than returning nothing for
    exactly the papers whose impact matters most to rank correctly.
    """
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT cohort_year, citation_count
            FROM paper_metrics
            WHERE citation_count IS NOT NULL AND cohort_year IS NOT NULL
            ORDER BY cohort_year, citation_count
            """
        ).fetchall()

    grouped: dict[int, list[int]] = {}
    for row in rows:
        grouped.setdefault(row["cohort_year"], []).append(row["citation_count"])

    return [
        Cohort(POOLED_CATEGORY, year, len(counts), _median(counts))
        for year, counts in sorted(grouped.items())
    ]


def cohort_for(
    category: str | None,
    year: int | None,
    cohorts: dict[tuple[str, int], Cohort],
) -> Cohort | None:
    """Best available peer group: the paper's own category-year, else the pooled year."""
    if year is None:
        return None
    own = cohorts.get((category, year)) if category else None
    if own is not None and own.is_reliable:
        return own
    return cohorts.get((POOLED_CATEGORY, year))


def normalized_impact(citations: int | None, cohort: Cohort | None) -> float | None:
    """Citations relative to the cohort median. None when it cannot be computed.

    A median of 0 (common for the newest cohorts, where most papers have no citations
    yet) would divide by zero, so those return None rather than infinity. Returning
    None is deliberate: it lets the ranker treat "unknown" as its own case instead of
    being handed a fabricated number it would take at face value.
    """
    if citations is None or cohort is None or not cohort.is_reliable:
        return None
    if cohort.median_citations <= 0:
        return None
    return citations / cohort.median_citations


def store_cohorts(cohorts: list[Cohort]) -> int:
    """Persists cohort statistics for reuse, keyed for direct lookup."""
    with db.connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cohort_stats (
                category TEXT NOT NULL,
                year INTEGER NOT NULL,
                size INTEGER NOT NULL,
                median_citations REAL NOT NULL,
                computed_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (category, year)
            );
            """
        )
        conn.executemany(
            "INSERT INTO cohort_stats (category, year, size, median_citations, computed_at) "
            "VALUES (?, ?, ?, ?, datetime('now')) "
            "ON CONFLICT(category, year) DO UPDATE SET "
            "size=excluded.size, median_citations=excluded.median_citations, computed_at=excluded.computed_at",
            [(c.category, c.year, c.size, c.median_citations) for c in cohorts],
        )
    return len(cohorts)


def load_cohorts() -> dict[tuple[str, int], Cohort]:
    """Cohort stats keyed by (category, year), or empty if never computed."""
    with db.connect() as conn:
        tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "cohort_stats" not in tables:
            return {}
        rows = conn.execute("SELECT category, year, size, median_citations FROM cohort_stats").fetchall()

    return {
        (r["category"], r["year"]): Cohort(r["category"], r["year"], r["size"], r["median_citations"])
        for r in rows
    }


def compute_and_store_cnci() -> int:
    """Computes field-normalized impact for every paper and stores it.

    Papers whose cohort is too small to normalize against keep a NULL cnci rather
    than a substitute value - "unknown" and "typical for its field" are different
    claims, and a ranker handed 1.0 for both would learn from a fabrication.
    """
    cohorts = load_cohorts()
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT paper_id, citation_count, cohort_year, primary_category FROM paper_metrics"
        ).fetchall()

        updates = []
        for row in rows:
            cohort = cohort_for(row["primary_category"], row["cohort_year"], cohorts)
            updates.append((normalized_impact(row["citation_count"], cohort), row["paper_id"]))

        conn.executemany("UPDATE paper_metrics SET cnci = ? WHERE paper_id = ?", updates)

    return sum(1 for value, _ in updates if value is not None)
