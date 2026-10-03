"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { deleteSession, listSessions, type ChatSession } from "@/lib/sessionStore";
import { relativeTime } from "@/lib/relativeTime";

/** Strips the most common markdown syntax for a plain-text preview line - a raw
 * "**Foo**" or "[1234.5678v1](...)" reads as noise in a one-line snippet. */
function stripMarkdown(text: string): string {
  return text
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/[*_`#>]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function KebabIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
      <circle cx="8" cy="3" r="1.4" />
      <circle cx="8" cy="8" r="1.4" />
      <circle cx="8" cy="13" r="1.4" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75">
      <path d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2m-8 0 .8 12.2a1 1 0 0 0 1 .8h6.4a1 1 0 0 0 1-.8L18 7" />
    </svg>
  );
}

function SessionRow({ session, onDelete }: { session: ChatSession; onDelete: (id: string) => void }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!menuOpen) return;
    function handlePointerDown(e: PointerEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenuOpen(false);
    }
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") setMenuOpen(false);
    }
    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKey);
    };
  }, [menuOpen]);

  const href =
    session.kind === "paper"
      ? `/paper/${encodeURIComponent(session.paperId ?? "")}?session=${session.id}`
      : `/search?session=${session.id}`;
  const lastTurn = session.turns[session.turns.length - 1];

  return (
    <li className="group relative flex items-center">
      <Link
        href={href}
        className="-mx-3 flex min-w-0 flex-1 flex-col gap-1 rounded-lg px-3 py-3.5 transition-colors hover:bg-surface-muted"
      >
        <div className="flex items-center gap-2">
          <span className="truncate text-[15px] font-medium text-foreground">{session.title}</span>
          <span className="shrink-0 text-[11px] text-foreground-muted/60">
            {session.kind === "paper" ? "paper" : "search"}
          </span>
        </div>
        <div className="flex min-w-0 gap-2 text-xs text-foreground-muted">
          <span className="shrink-0">{relativeTime(session.updatedAt)}</span>
          {lastTurn && (
            <>
              <span className="shrink-0">·</span>
              <span className="truncate">
                {lastTurn.role === "user" ? "" : "Agent: "}
                {stripMarkdown(lastTurn.content)}
              </span>
            </>
          )}
        </div>
      </Link>

      <div ref={menuRef} className="relative ml-1 shrink-0">
        <button
          onClick={() => setMenuOpen((v) => !v)}
          aria-label="Chat options"
          aria-expanded={menuOpen}
          className={`flex h-8 w-8 items-center justify-center rounded-full text-foreground-muted transition-colors hover:bg-surface-muted hover:text-foreground ${
            menuOpen ? "bg-surface-muted text-foreground" : "opacity-0 group-hover:opacity-100"
          }`}
        >
          <KebabIcon />
        </button>

        {menuOpen && (
          <div className="absolute top-full right-0 z-10 mt-1 w-36 overflow-hidden rounded-xl border border-border bg-surface py-1 shadow-[0_8px_24px_-8px_rgba(38,37,33,0.18)]">
            <button
              onClick={() => {
                setMenuOpen(false);
                onDelete(session.id);
              }}
              className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-red-600 transition-colors hover:bg-red-50"
            >
              <TrashIcon />
              Delete
            </button>
          </div>
        )}
      </div>
    </li>
  );
}

export default function HistoryPage() {
  const [sessions, setSessions] = useState<ChatSession[] | null>(null);

  useEffect(() => {
    setSessions(listSessions());
  }, []);

  function handleDelete(id: string) {
    deleteSession(id);
    setSessions(listSessions());
  }

  return (
    <main className="mx-auto flex w-full min-h-0 max-w-2xl flex-1 flex-col gap-1 overflow-y-auto px-6 py-8">
      <h1 className="font-serif text-2xl font-semibold text-foreground">History</h1>
      <p className="mb-6 text-sm text-foreground-muted">Every saved chat — search sessions and per-paper conversations.</p>

      {sessions === null ? null : sessions.length === 0 ? (
        <p className="text-sm text-foreground-muted">
          No chats yet. Start a{" "}
          <Link href="/search" className="text-accent hover:text-accent-hover">
            search
          </Link>{" "}
          to create one.
        </p>
      ) : (
        <ul className="flex flex-col divide-y divide-border pb-8">
          {sessions.map((session) => (
            <SessionRow key={session.id} session={session} onDelete={handleDelete} />
          ))}
        </ul>
      )}
    </main>
  );
}
