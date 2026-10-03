"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import MetricBadges from "@/components/MetricBadges";
import type { Paper, PaperMetrics } from "@/lib/api";
import { fetchMetrics } from "@/lib/metricsCache";
import { fetchPaper } from "@/lib/paperCache";

/** One paper in a result list.
 *
 * `min-w-0` on the card and on every text child is what actually contains the
 * content: a flex or grid item defaults to `min-width: auto`, meaning it refuses to
 * shrink below its contents, so a long title or venue pushes the card wider than its
 * column instead of wrapping or ellipsizing inside it. Without it, `truncate` and
 * `line-clamp` on the children are silently inert. */
export default function PaperCard({ paperId }: { paperId: string }) {
  const [paper, setPaper] = useState<Paper | null>(null);
  const [metrics, setMetrics] = useState<PaperMetrics | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchPaper(paperId)
      .then((p) => !cancelled && setPaper(p))
      .catch(() => {});
    // Metrics are deliberately a separate request that is allowed to fail quietly.
    // A paper the agent found live on arXiv has no scorecard yet, and that should
    // cost it a row of badges, not its card.
    fetchMetrics(paperId)
      .then((m) => !cancelled && setMetrics(m))
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [paperId]);

  const year = paper?.published?.slice(0, 4);
  const authors = paper
    ? paper.authors.slice(0, 2).join(", ") + (paper.authors.length > 2 ? " et al." : "")
    : "";

  return (
    <Link
      href={`/paper/${encodeURIComponent(paperId)}`}
      className="group flex min-w-0 flex-col gap-1.5 rounded-xl border border-border bg-surface p-3 transition-colors hover:border-accent"
    >
      <span className="line-clamp-2 min-w-0 text-sm leading-snug font-medium text-foreground group-hover:text-accent-hover">
        {paper?.title ?? paperId}
      </span>
      {paper && (
        <span className="flex min-w-0 items-baseline gap-1 text-xs text-foreground-muted">
          <span className="truncate">{authors || "Unknown authors"}</span>
          {year && <span className="shrink-0">· {year}</span>}
        </span>
      )}
      <MetricBadges metrics={metrics} />
    </Link>
  );
}
