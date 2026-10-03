"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Chat from "@/components/Chat";
import PaperCard from "@/components/PaperCard";
import StarterPrompts from "@/components/StarterPrompts";
import { canonicalPaperId } from "@/lib/formatMetrics";
import { search, type ChatTurn } from "@/lib/api";
import {
  deriveTitle,
  getLastSearchSessionId,
  getSession,
  newDraftSession,
  saveSession,
  setLastSearchSessionId,
  type ChatSession,
} from "@/lib/sessionStore";

function SearchPageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const sessionId = searchParams.get("session");

  const [session, setSession] = useState<ChatSession | null>(null);

  useEffect(() => {
    if (!sessionId) {
      // No explicit ?session= - this is what every plain "/search" link in the app
      // points at (the nav bar, a paper page's "back to search"), so resuming the
      // last active conversation here (rather than always starting fresh) is what
      // makes navigating away and back actually feel like coming back to the same
      // chat, no matter which link got you here.
      const lastId = getLastSearchSessionId();
      const resumed = lastId ? getSession(lastId) : undefined;
      const target = resumed ?? newDraftSession("search");
      router.replace(`/search?session=${target.id}`);
      return;
    }
    // A session id with no saved data yet is a draft that never got a first
    // message (e.g. a fresh visit, or a reload before sending anything) -
    // reconstruct an empty one with that same id rather than creating a new id.
    setSession(getSession(sessionId) ?? { ...newDraftSession("search"), id: sessionId });
    // Remember this as the session an unadorned "/search" link should resume.
    setLastSearchSessionId(sessionId);
  }, [sessionId, router]);

  function handleTurnsChange(turns: ChatTurn[]) {
    if (!session) return;
    const updated = saveSession({ ...session, turns, title: deriveTitle(session, turns) });
    setSession(updated);
  }

  function handleNewChat() {
    const draft = newDraftSession("search");
    router.push(`/search?session=${draft.id}`);
  }

  const paperIds = useMemo(() => {
    if (!session) return [];
    // Deduped by CANONICAL id, not the raw one. Some papers exist in the corpus under
    // both a bare and a versioned id and both are indexed, so a single answer can cite
    // `1810.04805` and `1810.04805v2` and the sidebar would list BERT twice - reading
    // as two findings where there is one paper.
    const seen = new Set<string>();
    const ids: string[] = [];
    for (const turn of session.turns) {
      for (const source of turn.sources ?? []) {
        const key = canonicalPaperId(source.paper_id);
        if (!seen.has(key)) {
          seen.add(key);
          ids.push(source.paper_id);
        }
      }
    }
    return ids;
  }, [session]);

  if (!session) return null;

  return (
    <main className="mx-auto flex w-full min-h-0 max-w-6xl flex-1 gap-8 px-6 py-8">
      <div className="flex min-h-0 flex-1 flex-col">
        <div className="mb-4 flex items-start justify-between gap-4">
          <div>
            <h1 className="font-serif text-2xl font-semibold text-foreground">Search</h1>
            <p className="text-sm text-foreground-muted">
              Ask a research question, then keep chatting — follow-ups can reference earlier papers.
            </p>
          </div>
          {session.turns.length > 0 && (
            <button
              onClick={handleNewChat}
              className="shrink-0 rounded-full border border-border px-3 py-1.5 text-xs font-medium text-foreground-muted transition-colors hover:border-accent hover:text-accent"
            >
              New chat
            </button>
          )}
        </div>
        {paperIds.length > 0 && (
          <details className="group mb-3 shrink-0 lg:hidden">
            <summary className="flex w-fit cursor-pointer list-none items-center gap-1 text-xs font-medium tracking-wide text-foreground-muted uppercase hover:text-accent [&::-webkit-details-marker]:hidden">
              Papers found ({paperIds.length})
              <span className="transition-transform group-open:rotate-90">›</span>
            </summary>
            <div className="scroll-area mt-2 flex max-h-56 flex-col gap-2 overflow-y-auto pr-2">
              {paperIds.map((id) => (
                <PaperCard key={id} paperId={id} />
              ))}
            </div>
          </details>
        )}
        <Chat
          key={session.id}
          turns={session.turns}
          onTurnsChange={handleTurnsChange}
          onSend={search}
          placeholder="e.g. how does retrieval improve factual accuracy in language models"
          emptyState={(send) => <StarterPrompts onPick={send} />}
        />
      </div>

      {/* Desktop: a persistent sidebar. Below lg there is no room for one, and the
          previous layout simply hid it - so on a laptop at 1280 the found papers were
          visible and on anything narrower they did not exist at all, with no hint that
          a panel had been dropped. A collapsible strip keeps them reachable instead of
          silently removing the only place metrics are shown. */}
      <aside className="hidden min-h-0 w-72 shrink-0 flex-col gap-3 lg:flex">
        <span className="shrink-0 text-xs font-medium tracking-wide text-foreground-muted uppercase">
          Papers found {paperIds.length > 0 && `(${paperIds.length})`}
        </span>
        <div className="scroll-area flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto pr-2">
          {paperIds.length === 0 && (
            <p className="text-sm text-foreground-muted">Papers surfaced by search will show up here.</p>
          )}
          {paperIds.map((id) => (
            <PaperCard key={id} paperId={id} />
          ))}
        </div>
      </aside>
    </main>
  );
}

export default function SearchPage() {
  return (
    <Suspense fallback={null}>
      <SearchPageInner />
    </Suspense>
  );
}
