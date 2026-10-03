/** Shared formatting for scientometric values.
 *
 * Shared because MetricBadges and Scorecard must not disagree: the same paper
 * showing "top 0.1%" on a card and "top 0.0%" on its detail page reads as a bug in
 * the measurement rather than in the formatter. */

/** Citation counts as compact labels: 193374 -> "193k", 847 -> "847". */
export function formatCount(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(n >= 10_000 ? 0 : 1)}k`;
  return String(n);
}

/** Field-normalized impact. These span six orders of magnitude - 0.4x for a paper
 * cited less than its peers, 6,446x for the transformer - so the precision has to
 * shrink as the number grows or the large values become unreadable. */
export function formatMultiple(x: number): string {
  if (x >= 100) return `${Math.round(x).toLocaleString()}x`;
  if (x >= 10) return `${x.toFixed(0)}x`;
  return `${x.toFixed(1)}x`;
}

/** Percentile as the phrase a reader actually parses.
 *
 * "99.9th percentile" and "top 0.1%" are the same fact; the second is the one that
 * lands. The floor at 0.1% exists because the top few papers of 8,770 all round to a
 * percentile of 100.0, and "top 0.0%" states that a non-empty set is empty. */
export function formatPercentile(percentile: number): string {
  const top = 100 - percentile;
  if (top < 0.1) return "top 0.1%";
  if (top < 1) return `top ${top.toFixed(1)}%`;
  if (percentile >= 95) return `top ${Math.round(top)}%`;
  return `${percentile.toFixed(0)}th pct`;
}

/** Strips an arXiv version suffix: `1810.04805v2` -> `1810.04805`.
 *
 * Mirrors `canonical_paper_id` on the backend. The corpus genuinely holds some papers
 * under both spellings, and retrieval can return each as a separate result, so any
 * list of papers shown to a user has to collapse them or the same paper appears
 * twice - which reads as the system having found two papers when it found one. */
export function canonicalPaperId(paperId: string): string {
  return paperId.replace(/v\d+$/, "");
}
