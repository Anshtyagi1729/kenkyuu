"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { fetchPaper, peekPaper } from "@/lib/paperCache";

/** An inline numbered citation: a small "[2]" that opens the paper, with its title on
 * hover. Replaces the raw ids answers used to show ("1810.04805v2", or a 40-character
 * hash for a journal paper), which identified a paper only to someone who already
 * knew it. The title is looked up through the shared paper cache, so the many chips in
 * one answer cost one request per paper rather than one per chip. */
export default function CitationChip({ number, href }: { number: string; href: string }) {
  const paperId = decodeURIComponent(href.replace(/^\/paper\//, ""));
  const [title, setTitle] = useState<string | null>(() => peekPaper(paperId)?.title ?? null);

  useEffect(() => {
    if (title) return;
    let cancelled = false;
    fetchPaper(paperId).then((p) => {
      if (!cancelled && p) setTitle(p.title);
    });
    return () => {
      cancelled = true;
    };
  }, [paperId, title]);

  return (
    <Link
      href={href}
      title={title ?? paperId}
      aria-label={`Source ${number}${title ? `: ${title}` : ""}`}
      className="relative -top-[0.1em] mx-[1px] inline-flex h-[1.25em] min-w-[1.25em] items-center justify-center rounded-md bg-accent-soft px-[0.3em] text-[0.72em] leading-none font-semibold text-accent-hover no-underline transition-colors hover:bg-accent hover:text-white"
    >
      {number}
    </Link>
  );
}
