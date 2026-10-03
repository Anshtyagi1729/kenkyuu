import { getMetricsBatch, type PaperMetrics } from "@/lib/api";

/** Coalesces per-card metric lookups into one batched request.
 *
 * A result list renders one PaperCard per source, and each card independently wants
 * its own scorecard. Fetching per card turns a ten-source answer into ten round
 * trips that all arrive at the same endpoint a millisecond apart. This collects
 * every id requested within the same tick and sends them as a single POST, which is
 * what /papers/metrics exists to serve.
 *
 * Results are also cached for the page's lifetime. The same paper routinely appears
 * in several answers in one session, and its citation counts do not change between
 * two renders a few seconds apart. */

const cache = new Map<string, PaperMetrics | null>();
const inFlight = new Map<string, Promise<PaperMetrics | null>>();

let pending: string[] = [];
let scheduled: Promise<void> | null = null;

async function flush(): Promise<void> {
  const ids = pending;
  pending = [];
  scheduled = null;
  if (ids.length === 0) return;

  try {
    const results = await getMetricsBatch(ids);
    const found = new Map(results.map((m) => [m.paper_id, m]));
    // Every requested id is written, not just the ones that came back. A paper the
    // corpus does not hold must be cached as a definite null, or each re-render
    // would queue it again and retry forever.
    for (const id of ids) cache.set(id, found.get(id) ?? null);
  } catch {
    // A failed batch is cached as nothing at all rather than as null, so a transient
    // network error does not permanently mark every paper in it as having no metrics.
    for (const id of ids) cache.delete(id);
  } finally {
    for (const id of ids) inFlight.delete(id);
  }
}

export function fetchMetrics(paperId: string): Promise<PaperMetrics | null> {
  if (cache.has(paperId)) return Promise.resolve(cache.get(paperId) ?? null);

  const existing = inFlight.get(paperId);
  if (existing) return existing;

  if (!pending.includes(paperId)) pending.push(paperId);
  scheduled ??= Promise.resolve().then(flush);

  const promise = scheduled.then(() => cache.get(paperId) ?? null);
  inFlight.set(paperId, promise);
  return promise;
}
