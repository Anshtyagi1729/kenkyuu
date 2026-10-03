"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";
import CitationGraph from "@/components/CitationGraph";
import { type Paper } from "@/lib/api";
import { fetchPaper } from "@/lib/paperCache";

export default function CitationGraphPage({ params }: PageProps<"/paper/[id]/graph">) {
  const { id } = use(params);
  const paperId = decodeURIComponent(id);

  const [loaded, setLoaded] = useState<{ id: string; paper: Paper | null; error: string | null } | null>(null);

  // Paper data is stored together with the id it belongs to, instead of being cleared
  // synchronously at the top of the effect. A synchronous setState inside an effect
  // cascades an extra render, and clearing-then-refetching also flashes an empty
  // state on every navigation between papers. Comparing the stored id against the
  // current one answers "is this the right paper's data" without either.
  useEffect(() => {
    let cancelled = false;
    fetchPaper(paperId)
      .then((p) =>
        !cancelled &&
        setLoaded({ id: paperId, paper: p, error: p ? null : `Paper ${paperId} not found.` }),
      )
      .catch((err) => !cancelled && setLoaded({ id: paperId, paper: null, error: err.message }));
    return () => {
      cancelled = true;
    };
  }, [paperId]);

  const current = loaded?.id === paperId ? loaded : null;
  const paper = current?.paper ?? null;
  const error = current?.error ?? null;

  return (
    <main className="mx-auto flex w-full min-h-0 max-w-5xl flex-1 flex-col gap-4 px-6 py-8">
      <Link
        href={`/paper/${encodeURIComponent(paperId)}`}
        className="shrink-0 text-sm text-foreground-muted hover:text-accent"
      >
        ← Back to paper
      </Link>

      <div className="shrink-0">
        <h1 className="font-serif text-xl leading-snug font-semibold text-foreground">
          {paper?.title ?? (error ? "Citation graph" : "Loading…")}
        </h1>
        <p className="text-sm text-foreground-muted">
          References and citing papers. Click a node for details, then Expand to pull in its neighbors too.
        </p>
      </div>

      {error && <p className="shrink-0 text-sm text-red-600">{error}</p>}

      <CitationGraph centerPaperId={paperId} />
    </main>
  );
}
