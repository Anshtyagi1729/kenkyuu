"use client";

import { useState } from "react";
import { compare, type CompareResponse } from "@/lib/api";
import MarkdownAnswer from "@/components/MarkdownAnswer";
import PaperPicker from "@/components/PaperPicker";
import Link from "next/link";

export default function ComparePage() {
  const [picked, setPicked] = useState<string[]>([]);
  const [paperIdsInput, setPaperIdsInput] = useState("");
  const [focus, setFocus] = useState("");
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Picked papers and manually typed ids are unioned rather than one overriding the
  // other, so a user can compare two papers from their library against one they
  // happen to have an id for without the two inputs fighting.
  const manualIds = paperIdsInput
    .split(",")
    .map((id) => id.trim())
    .filter(Boolean);
  const paperIds = [...new Set([...picked, ...manualIds])];

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (paperIds.length < 2 || loading) return;

    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await compare(paperIds, focus.trim() || undefined));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-6 overflow-y-auto px-6 py-8">
      <div>
        <h1 className="font-serif text-2xl font-semibold text-foreground">Compare papers</h1>
        <p className="text-sm text-foreground-muted">
          Pick two or more papers to compare methodology, results, and claims.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="flex flex-col gap-3">
        <div className="flex flex-col gap-2">
          <span className="text-xs font-medium tracking-wide text-foreground-muted uppercase">
            Your papers {paperIds.length > 0 && `· ${paperIds.length} selected`}
          </span>
          <PaperPicker selected={picked} onChange={setPicked} />
        </div>

        <details className="group">
          <summary className="flex w-fit cursor-pointer list-none items-center gap-1 text-xs text-foreground-muted hover:text-accent [&::-webkit-details-marker]:hidden">
            Or paste arXiv IDs
            <span className="transition-transform group-open:rotate-90">›</span>
          </summary>
          <input
            value={paperIdsInput}
            onChange={(e) => setPaperIdsInput(e.target.value)}
            placeholder="e.g. 2005.11401v4, 2312.10997v5"
            className="mt-2 w-full rounded-full border border-border bg-surface px-4 py-2.5 text-[15px] outline-none transition-shadow focus:border-accent focus:ring-4 focus:ring-accent/10"
          />
        </details>
        <input
          value={focus}
          onChange={(e) => setFocus(e.target.value)}
          placeholder="Optional focus, e.g. evaluation methodology (default: methodology, results, contributions)"
          className="rounded-full border border-border bg-surface px-4 py-2.5 text-[15px] outline-none transition-shadow focus:border-accent focus:ring-4 focus:ring-accent/10"
        />
        <button
          type="submit"
          disabled={loading || paperIds.length < 2}
          className="self-start rounded-full bg-accent px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-accent-hover disabled:opacity-40"
        >
          {loading ? "Comparing…" : paperIds.length < 2 ? "Select 2 or more papers" : `Compare ${paperIds.length} papers`}
        </button>
      </form>

      {error && <p className="text-sm text-red-600">{error}</p>}

      {result && (
        <div className="flex flex-col gap-6 pb-8">
          <article className="rounded-2xl border border-border bg-surface p-5">
            <MarkdownAnswer text={result.comparison} />
          </article>

          {result.sources.length > 0 && (
            <div className="flex flex-col gap-2">
              <h2 className="text-xs font-medium tracking-wide text-foreground-muted uppercase">
                Sources ({result.sources.length})
              </h2>
              <ul className="flex flex-col gap-2">
                {result.sources.map((source) => (
                  <li
                    key={source.chunk_id}
                    className="rounded-xl border border-border bg-surface p-3 text-xs leading-relaxed text-foreground-muted"
                  >
                    <Link
                      href={`/paper/${encodeURIComponent(source.paper_id)}`}
                      className="mr-1 font-medium text-accent hover:text-accent-hover"
                    >
                      [{source.paper_id}
                      {source.start_page != null &&
                        ` p.${source.start_page === source.end_page ? source.start_page : `${source.start_page}-${source.end_page}`}`}
                      ]
                    </Link>
                    {source.text}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </main>
  );
}
