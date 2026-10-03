"use client";

import type { PaperMetrics } from "@/lib/api";
import { formatCount, formatMultiple, formatPercentile } from "@/lib/formatMetrics";
import { abbreviateVenue } from "@/lib/venues";

/** Compact impact badges for a paper in a result list.
 *
 * The point of the badges is to make a recommendation checkable. "This paper is
 * influential" is an opinion; "6,446x the 2017 field median" is a measurement the
 * reader can disagree with. So the badges show the normalized figure first and the
 * raw count second - a raw citation count is mostly a proxy for age, and putting it
 * first invites exactly the comparison between a 2017 and a 2025 paper that the
 * normalization exists to prevent.
 *
 * A metric that is null is OMITTED, never defaulted. There is no badge that means
 * "we could not compute this", because a badge reading 0 or "-" in a row of real
 * numbers reads as a measured low value.
 *
 * On overflow: these render inside a 288px sidebar card. Numeric badges are short and
 * fixed-width enough to stay whole, but venue names from Semantic Scholar run past 70
 * characters, so the venue badge is allowed to shrink and ellipsize while the numbers
 * never are. Its full name is in the tooltip either way. */

/** Only the extreme tail is worth a badge. A paper at the 60th centrality percentile
 * is unremarkable, and labelling it would turn the badge row into noise on every
 * result rather than a signal on the few that earn it. */
const CENTRALITY_BADGE_THRESHOLD = 95;

type Tone = "accent" | "plain";

function Badge({
  label,
  title,
  tone = "plain",
  shrink = false,
}: {
  label: string;
  title: string;
  tone?: Tone;
  /** Allow this badge to be squeezed and ellipsized when the row is too narrow.
   * Off by default: a truncated "6,4…x field median" is worse than useless, because
   * a number that has lost its magnitude still reads as a number. */
  shrink?: boolean;
}) {
  const toneClass =
    tone === "accent"
      ? "border-accent/40 bg-accent-soft text-accent-hover"
      : "border-border bg-surface-muted text-foreground-muted";
  // min-w-0 is what actually lets a flex item shrink below its content width; without
  // it, `truncate` has no effect inside a flex row and the badge overflows the card.
  const sizeClass = shrink ? "min-w-0 max-w-full shrink truncate" : "shrink-0 whitespace-nowrap";
  return (
    <span
      title={title}
      className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] leading-none font-medium ${sizeClass} ${toneClass}`}
    >
      {label}
    </span>
  );
}

export default function MetricBadges({
  metrics,
  className = "",
}: {
  metrics: PaperMetrics | null;
  className?: string;
}) {
  if (!metrics) return null;

  const badges: React.ReactNode[] = [];

  if (metrics.cnci !== null) {
    const field = metrics.primary_category ?? "all fields";
    badges.push(
      <Badge
        key="cnci"
        tone="accent"
        label={`${formatMultiple(metrics.cnci)} field median`}
        title={`Cited ${formatMultiple(metrics.cnci)} as often as the median ${metrics.cohort_year ?? ""} paper in ${field}. Field-normalized so papers of different ages stay comparable.`}
      />,
    );
  }

  if (metrics.citation_count !== null) {
    badges.push(
      <Badge
        key="cites"
        label={`${formatCount(metrics.citation_count)} cites`}
        title={
          metrics.influential_citation_count !== null
            ? `${metrics.citation_count.toLocaleString()} citations, of which ${metrics.influential_citation_count.toLocaleString()} are judged substantive rather than passing references.`
            : `${metrics.citation_count.toLocaleString()} citations.`
        }
      />,
    );
  }

  if (metrics.pagerank_percentile !== null && metrics.pagerank_percentile >= CENTRALITY_BADGE_THRESHOLD) {
    const label = formatPercentile(metrics.pagerank_percentile);
    badges.push(
      <Badge
        key="central"
        tone="accent"
        label={`${label} central`}
        title={`Citation-network centrality (PageRank) in the ${label} of this corpus${
          metrics.in_degree ? `, cited by ${metrics.in_degree.toLocaleString()} corpus papers` : ""
        }. Measures being cited BY influential papers, not citation volume.`}
      />,
    );
  }

  const venue = abbreviateVenue(metrics.venue);
  if (venue) {
    badges.push(
      <Badge
        key="venue"
        label={venue.short}
        shrink={!venue.abbreviated}
        title={venue.abbreviated ? `${venue.short} — ${venue.full}` : `Published at ${venue.full}`}
      />,
    );
  }

  if (badges.length === 0) return null;

  return <div className={`flex min-w-0 flex-wrap items-center gap-1.5 ${className}`}>{badges}</div>;
}
