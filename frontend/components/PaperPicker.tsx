"use client";

import { useEffect, useState } from "react";
import { listFolders, getFolderPapers, type Paper } from "@/lib/api";
import { fetchPaper } from "@/lib/paperCache";
import { recentPaperIds } from "@/lib/sessionStore";

/** Multi-select over papers the user has actually encountered.
 *
 * Replaces asking them to type comma-separated arXiv ids from memory. Nobody
 * remembers that 1810.04805 is BERT; requiring it made the compare page usable only
 * by someone who already had the ids on a clipboard, which is to say not usable.
 *
 * Candidates come from two places that need no new storage: papers saved to the
 * library, and papers cited in previous answers (already recorded in each session).
 * Manual entry stays available for a paper the user has neither saved nor seen. */
export default function PaperPicker({
  selected,
  onChange,
  max = 5,
}: {
  selected: string[];
  onChange: (ids: string[]) => void;
  max?: number;
}) {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const ids: string[] = [];
      const seen = new Set<string>();
      const add = (id: string) => {
        const key = id.replace(/v\d+$/, "");
        if (!seen.has(key)) {
          seen.add(key);
          ids.push(id);
        }
      };

      // Saved papers first - an explicit save is a stronger signal of intent than
      // having once appeared in a source list.
      try {
        const folders = await listFolders();
        const perFolder = await Promise.all(folders.map((f) => getFolderPapers(f.folder_id)));
        for (const folderPapers of perFolder) for (const p of folderPapers) add(p.paper_id);
      } catch {
        // Library unavailable - recents alone still make the picker useful.
      }
      for (const id of recentPaperIds()) add(id);

      const resolved = await Promise.all(
        ids.slice(0, 40).map((id) => fetchPaper(id)),
      );
      if (cancelled) return;
      setPapers(resolved.filter((p): p is Paper => p !== null));
      setLoading(false);
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  function toggle(paperId: string) {
    if (selected.includes(paperId)) {
      onChange(selected.filter((id) => id !== paperId));
    } else if (selected.length < max) {
      onChange([...selected, paperId]);
    }
  }

  if (loading) {
    return <p className="text-sm text-foreground-muted">Loading your papers…</p>;
  }

  if (papers.length === 0) {
    return (
      <p className="text-sm text-foreground-muted">
        No papers yet — search for some first, or paste arXiv IDs below.
      </p>
    );
  }

  return (
    <div className="scroll-area flex max-h-72 flex-col gap-1.5 overflow-y-auto pr-2">
      {papers.map((paper) => {
        const isSelected = selected.includes(paper.paper_id);
        const atLimit = !isSelected && selected.length >= max;
        return (
          <button
            key={paper.paper_id}
            type="button"
            onClick={() => toggle(paper.paper_id)}
            disabled={atLimit}
            className={`flex min-w-0 items-start gap-2.5 rounded-xl border p-2.5 text-left transition-colors ${
              isSelected
                ? "border-accent bg-accent-soft"
                : "border-border bg-surface hover:border-accent disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:border-border"
            }`}
          >
            <span
              aria-hidden
              className={`mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded border ${
                isSelected ? "border-accent bg-accent text-white" : "border-border bg-surface"
              }`}
            >
              {isSelected && (
                <svg viewBox="0 0 12 12" className="h-3 w-3">
                  <path
                    d="M2.5 6.2l2.2 2.3L9.5 3.7"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.8"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              )}
            </span>
            <span className="flex min-w-0 flex-col gap-0.5">
              <span className="line-clamp-2 text-sm leading-snug text-foreground">{paper.title}</span>
              <span className="truncate text-xs text-foreground-muted">
                {paper.authors.slice(0, 2).join(", ")}
                {paper.authors.length > 2 ? " et al." : ""}
                {paper.published ? ` · ${paper.published.slice(0, 4)}` : ""}
              </span>
            </span>
          </button>
        );
      })}
    </div>
  );
}
