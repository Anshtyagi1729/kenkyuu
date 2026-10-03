import { getPaper, type Paper } from "@/lib/api";

/** Deduped, cached lookups for paper metadata.
 *
 * The same paper is requested by several independent components on one screen - a
 * card in the desktop sidebar, the same card in the collapsed mobile list, a source
 * pill wanting its title - and React's development StrictMode invokes every effect
 * twice on top of that. Measured on one answer: seven papers produced twenty-eight
 * identical GET /papers/{id} requests.
 *
 * Each component owning its own fetch is the right shape; what was missing is a
 * place for them to agree that the request is already happening. An in-flight map
 * collapses concurrent callers onto one promise, and the result is kept for the
 * page's lifetime because a paper's title and authors do not change while someone
 * reads about it.
 *
 * Failures are deliberately NOT cached. A paper missing because the backend was
 * briefly down should be retried on the next render, not remembered as absent for
 * the rest of the session. */

const cache = new Map<string, Paper>();
const inFlight = new Map<string, Promise<Paper | null>>();

export function fetchPaper(paperId: string): Promise<Paper | null> {
  const hit = cache.get(paperId);
  if (hit) return Promise.resolve(hit);

  const existing = inFlight.get(paperId);
  if (existing) return existing;

  const promise = getPaper(paperId)
    .then((paper) => {
      cache.set(paperId, paper);
      return paper;
    })
    .catch(() => null)
    .finally(() => {
      inFlight.delete(paperId);
    });

  inFlight.set(paperId, promise);
  return promise;
}

/** Synchronous peek, for a component that can render a fallback and fill in later. */
export function peekPaper(paperId: string): Paper | undefined {
  return cache.get(paperId);
}
