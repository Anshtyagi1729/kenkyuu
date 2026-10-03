"use client";

import { Suspense, use, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import Chat from "@/components/Chat";
import SaveButton from "@/components/SaveButton";
import Scorecard from "@/components/Scorecard";
import { chatWithPaper, type ChatTurn, type Paper } from "@/lib/api";
import { fetchPaper } from "@/lib/paperCache";
import {
  findLatestPaperSession,
  getSession,
  newDraftSession,
  saveSession,
  type ChatSession,
} from "@/lib/sessionStore";

function PaperPageInner({ params }: PageProps<"/paper/[id]">) {
  const { id } = use(params);
  const paperId = decodeURIComponent(id);
  const router = useRouter();
  const searchParams = useSearchParams();
  const sessionId = searchParams.get("session");

  const [loaded, setLoaded] = useState<{ id: string; paper: Paper | null; error: string | null } | null>(null);
  const [session, setSession] = useState<ChatSession | null>(null);

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

  // Resolve which session to show: an explicit ?session= always wins (found, or
  // reconstructed as an empty draft with that same id - never silently swapped for
  // a different one). With no ?session=, continue the most recent conversation
  // about this paper if one exists, else start a brand new draft.
  useEffect(() => {
    if (sessionId) {
      const existing = getSession(sessionId);
      setSession(
        existing ?? { ...newDraftSession("paper", { paperId, paperTitle: paper?.title ?? paperId }), id: sessionId },
      );
      return;
    }
    const latest = findLatestPaperSession(paperId);
    if (latest) {
      router.replace(`/paper/${encodeURIComponent(paperId)}?session=${latest.id}`);
      return;
    }
    if (!paper) return;
    const draft = newDraftSession("paper", { paperId, paperTitle: paper.title });
    router.replace(`/paper/${encodeURIComponent(paperId)}?session=${draft.id}`);
  }, [sessionId, paperId, paper, router]);

  function handleTurnsChange(turns: ChatTurn[]) {
    if (!session) return;
    setSession(saveSession({ ...session, turns }));
  }

  function handleNewChat() {
    const draft = newDraftSession("paper", { paperId, paperTitle: paper?.title ?? paperId });
    router.push(`/paper/${encodeURIComponent(paperId)}?session=${draft.id}`);
  }

  return (
    <main className="mx-auto flex w-full min-h-0 max-w-3xl flex-1 flex-col gap-4 px-6 py-8">
      <Link href="/search" className="shrink-0 text-sm text-foreground-muted hover:text-accent">
        ← Back to search
      </Link>

      {error && <p className="shrink-0 text-sm text-red-600">{error}</p>}

      {paper && (
        <div className="flex shrink-0 flex-col gap-2 rounded-2xl border border-border bg-surface p-5">
          <div className="flex items-start justify-between gap-4">
            <h1 className="font-serif text-xl leading-snug font-semibold text-foreground">{paper.title}</h1>
            <div className="flex shrink-0 items-center gap-2">
              <Link
                href={`/paper/${encodeURIComponent(paperId)}/graph`}
                className="rounded-full border border-border px-3 py-1.5 text-xs font-medium text-foreground-muted transition-colors hover:border-accent hover:text-accent"
              >
                Citation graph
              </Link>
              <SaveButton paperId={paperId} />
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-foreground-muted">
            <span>{paper.authors.join(", ") || "Unknown authors"}</span>
            {paper.published && <span>· {paper.published.slice(0, 10)}</span>}
            {paper.external_ids?.ArXiv && (
              <a
                href={`https://arxiv.org/abs/${paper.external_ids.ArXiv}`}
                target="_blank"
                rel="noreferrer"
                className="text-accent hover:text-accent-hover"
              >
                · arXiv:{paper.external_ids.ArXiv}
              </a>
            )}
          </div>
          <Scorecard paperId={paperId} />
          {paper.abstract && (
            <details className="group mt-1">
              <summary className="flex w-fit cursor-pointer list-none items-center gap-1 text-xs font-medium text-foreground-muted hover:text-accent [&::-webkit-details-marker]:hidden">
                Abstract
                <span className="transition-transform group-open:rotate-90">›</span>
              </summary>
              <p className="scroll-area mt-2 max-h-48 overflow-y-auto pr-3 text-sm leading-relaxed text-foreground-muted">
                {paper.abstract}
              </p>
            </details>
          )}
        </div>
      )}

      {session && (
        <div className="flex min-h-0 flex-1 flex-col">
          <div className="mb-3 flex shrink-0 items-center justify-between">
            <h2 className="text-xs font-medium tracking-wide text-foreground-muted uppercase">
              Chat about this paper
            </h2>
            {session.turns.length > 0 && (
              <button
                onClick={handleNewChat}
                className="rounded-full border border-border px-3 py-1.5 text-xs font-medium text-foreground-muted transition-colors hover:border-accent hover:text-accent"
              >
                New chat
              </button>
            )}
          </div>
          <Chat
            key={session.id}
            turns={session.turns}
            onTurnsChange={handleTurnsChange}
            onSend={(history, onEvent, signal) => chatWithPaper(paperId, history, onEvent, signal)}
            placeholder="Ask something about this paper"
            emptyState={
              <p className="py-6 text-sm text-foreground-muted">
                Ask about methodology, results, limitations — answers are grounded in this paper&apos;s full text.
              </p>
            }
          />
        </div>
      )}
    </main>
  );
}

export default function PaperPage(props: PageProps<"/paper/[id]">) {
  return (
    <Suspense fallback={null}>
      <PaperPageInner {...props} />
    </Suspense>
  );
}
