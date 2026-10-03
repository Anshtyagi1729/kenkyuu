"use client";

import { useEffect, useState } from "react";
import type { PaperMetrics } from "@/lib/api";
import { formatMultiple, formatPercentile } from "@/lib/formatMetrics";
import { fetchMetrics } from "@/lib/metricsCache";

/** The full scientometric scorecard for one paper.
 *
 * Where MetricBadges gives a glance, this gives the reading. Each figure is paired
 * with what it is normalized against, because the numbers are uninterpretable
 * without that: 193,374 citations means little until you know the median 2017 paper
 * in the same field has 30, and a PageRank of 0.0198 means nothing at all until it
 * is placed in a distribution.
 *
 * Metrics that could not be computed are shown as an explicit explanation rather
 * than hidden or zeroed. "Too few 2015 papers in the corpus to establish a median"
 * is a more useful thing for a professor to read than a silently missing row, and
 * far better than a fabricated 0. */

function Stat({
  label,
  value,
  detail,
  missing,
}: {
  label: string;
  value: string | null;
  detail: string;
  missing?: string;
}) {
  return (
    <div className="flex flex-col gap-0.5 rounded-lg border border-border bg-surface px-3 py-2.5">
      <span className="text-[11px] font-medium tracking-wide text-foreground-muted uppercase">{label}</span>
      {value !== null ? (
        <>
          <span className="font-serif text-lg leading-tight font-semibold text-foreground">{value}</span>
          <span className="text-[11px] leading-snug text-foreground-muted">{detail}</span>
        </>
      ) : (
        <span className="mt-0.5 text-[11px] leading-snug text-foreground-muted italic">
          {missing ?? "Not computed"}
        </span>
      )}
    </div>
  );
}

export default function Scorecard({ paperId }: { paperId: string }) {
  // The loaded paper id is stored ALONGSIDE its metrics rather than in a separate
  // "loaded" flag reset at the top of the effect. Resetting a flag synchronously
  // inside an effect triggers a cascading render, and it is also subtly wrong:
  // between the reset and the fetch resolving there is a frame where stale metrics
  // are paired with a new paperId. Comparing the stored id to the current prop
  // answers "is this data for this paper" directly, with no extra state.
  const [result, setResult] = useState<{ id: string; metrics: PaperMetrics | null } | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchMetrics(paperId)
      .then((m) => !cancelled && setResult({ id: paperId, metrics: m }))
      .catch(() => !cancelled && setResult({ id: paperId, metrics: null }));
    return () => {
      cancelled = true;
    };
  }, [paperId]);

  const loaded = result?.id === paperId;
  const metrics = loaded ? result.metrics : null;

  // Nothing at all rather than an empty shell: a paper discovered live from arXiv
  // and never ingested has no scorecard, and an "Impact" heading over four "not
  // computed" boxes would imply we measured something and found it lacking.
  if (!loaded || !metrics) return null;

  const {
    citation_count,
    influential_citation_count,
    cnci,
    cohort_year,
    primary_category,
    pagerank_percentile,
    in_degree,
    venue,
  } = metrics;

  const field = primary_category ?? "all fields";

  return (
    <details className="group" open>
      <summary className="flex w-fit cursor-pointer list-none items-center gap-1 text-xs font-medium text-foreground-muted hover:text-accent [&::-webkit-details-marker]:hidden">
        Impact
        <span className="transition-transform group-open:rotate-90">›</span>
      </summary>

      <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat
          label="Field-normalized"
          value={cnci !== null ? formatMultiple(cnci) : null}
          detail={`the median ${cohort_year ?? ""} paper in ${field}`}
          missing={
            cohort_year
              ? `Too few ${cohort_year} papers in the corpus to establish a reliable median`
              : "No submission year on record"
          }
        />
        <Stat
          label="Citations"
          value={citation_count !== null ? citation_count.toLocaleString() : null}
          detail={
            influential_citation_count !== null
              ? `${influential_citation_count.toLocaleString()} judged substantive`
              : "total, all sources"
          }
          missing="Not indexed by Semantic Scholar"
        />
        <Stat
          label="Network centrality"
          value={pagerank_percentile !== null ? formatPercentile(pagerank_percentile) : null}
          detail={
            in_degree
              ? `PageRank; cited by ${in_degree.toLocaleString()} corpus papers`
              : "PageRank; no citing papers inside this corpus"
          }
          missing="Not present in the citation graph"
        />
        <Stat label="Venue" value={venue} detail="publication record" missing="Preprint, or venue unknown" />
      </div>

      <p className="mt-2 text-[11px] leading-relaxed text-foreground-muted">
        Percentiles are against this corpus (deep learning, 2017&ndash;2026), not all of science.
        Field normalization divides by the median citations of papers from the same field and year,
        so papers of different ages stay comparable &mdash; a raw count mostly measures age.
      </p>
    </details>
  );
}
