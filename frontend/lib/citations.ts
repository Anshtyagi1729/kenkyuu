import { canonicalPaperId } from "@/lib/formatMetrics";

/** Turning the paper ids an answer cites into numbered citations.
 *
 * The model cites papers by id, and ids come in three forms: arXiv ids
 * ("1810.04805v2"), OpenAlex work ids ("W4391234567", journal papers recorded from
 * OpenAlex) and Semantic Scholar hashes (40 hex characters). It also wraps them
 * inconsistently: [id], 【id】, [id1, id2], "arXiv:id", or bare inside table cells.
 * Only bare and square-bracketed arXiv ids used to be recognised, so every other form
 * reached the reader as a raw id - the panel's own paper was cited as a 40-character
 * hash.
 *
 * Citations are numbered by first appearance, and the Sources list below the answer
 * uses the same numbering, so [2] in the text and 2 in the list are the same paper. */

const ID = String.raw`(?:\d{4}\.\d{4,5}(?:v\d+)?|W\d{7,}|[0-9a-f]{40})`;

// A bracketed group: [id], 【id】, [id1, id2; id3].
const GROUP = String.raw`[\[【]\s*(?:arXiv[:\s]\s*)?${ID}(?:\s*[,;]\s*(?:arXiv[:\s]\s*)?${ID})*\s*[\]】]`;
// A bare id, optionally prefixed "arXiv:" or "arXiv ", that is not part of a URL, path or word.
const BARE = String.raw`(?<![\w/.:#=-])(?:arXiv[:\s]\s*)?${ID}(?![\w/.])`;

const CITATION = new RegExp(`${GROUP}|${BARE}`, "g");
const ID_ONLY = new RegExp(ID, "g");

// Spans that must be left alone: existing markdown links (an id inside a URL is part
// of the URL), inline code, and math.
const PROTECTED = /\[[^\]\n]*\]\([^)\n]*\)|`[^`\n]*`|\$\$[\s\S]*?\$\$|\$[^$\n]*\$/g;

/** Canonical ids in order of first citation. */
export function citationOrder(text: string): string[] {
  const order: string[] = [];
  forEachUnprotected(text, (segment) => {
    for (const match of segment.match(CITATION) ?? []) {
      for (const id of match.match(ID_ONLY) ?? []) {
        const canonical = canonicalPaperId(id);
        if (!order.includes(canonical)) order.push(canonical);
      }
    }
    return segment;
  });
  return order;
}

/** Rewrites citations as markdown links whose text is the citation number. The href is
 * the paper page; the link renderer recognises it and draws a citation chip. */
export function linkifyCitations(text: string): string {
  const order = citationOrder(text);
  return forEachUnprotected(text, (segment) =>
    segment.replace(CITATION, (match) =>
      (match.match(ID_ONLY) ?? [])
        .map((id) => `[${order.indexOf(canonicalPaperId(id)) + 1}](/paper/${encodeURIComponent(id)})`)
        .join(""),
    ),
  );
}

/** Applies `transform` to the text between protected spans, leaving those spans as is. */
function forEachUnprotected(text: string, transform: (segment: string) => string): string {
  let out = "";
  let last = 0;
  for (const match of text.matchAll(PROTECTED)) {
    out += transform(text.slice(last, match.index)) + match[0];
    last = (match.index ?? 0) + match[0].length;
  }
  return out + transform(text.slice(last));
}
