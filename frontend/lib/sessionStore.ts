"use client";

import type { ChatTurn } from "@/lib/api";

export interface ChatSession {
  id: string;
  kind: "search" | "paper";
  paperId?: string;
  paperTitle?: string;
  title: string;
  turns: ChatTurn[];
  createdAt: number;
  updatedAt: number;
}

const STORAGE_KEY = "paper-agent:sessions";
const LAST_SEARCH_SESSION_KEY = "paper-agent:last-search-session";
const TITLE_MAX_CHARS = 60;

/** The most recently active /search session id, so "back to search" from a paper
 * page can return to the actual conversation you came from instead of always
 * landing on a brand new empty one - deliberately not based on browser history
 * (unreliable: doesn't exist for a middle-clicked new tab, a bookmark, etc). */
export function getLastSearchSessionId(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(LAST_SEARCH_SESSION_KEY);
  } catch {
    return null;
  }
}

export function setLastSearchSessionId(id: string): void {
  try {
    window.localStorage.setItem(LAST_SEARCH_SESSION_KEY, id);
  } catch {
    // storage disabled/full - "back to search" just falls back to a fresh chat
  }
}

function readAll(): ChatSession[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as ChatSession[]) : [];
  } catch {
    return [];
  }
}

function writeAll(sessions: ChatSession[]): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
  } catch {
    // storage disabled/full - degrade to no persistence for this write
  }
}

export function listSessions(): ChatSession[] {
  return readAll().sort((a, b) => b.updatedAt - a.updatedAt);
}

export function getSession(id: string): ChatSession | undefined {
  return readAll().find((s) => s.id === id);
}

/** Most recently updated existing session for a given paper, if any. */
export function findLatestPaperSession(paperId: string): ChatSession | undefined {
  return listSessions().find((s) => s.kind === "paper" && s.paperId === paperId);
}

/** Builds a new session object but does NOT persist it - a session should only
 * show up in history once it actually has a message, not the moment a page is
 * visited. Give it to the UI immediately (for a stable id/URL) and pass it through
 * saveSession() the first time a real turn is added; saveSession's upsert handles
 * the rest transparently. */
export function newDraftSession(kind: "search" | "paper", opts?: { paperId?: string; paperTitle?: string }): ChatSession {
  const now = Date.now();
  return {
    id: `${kind}-${now}-${Math.random().toString(36).slice(2, 8)}`,
    kind,
    paperId: opts?.paperId,
    paperTitle: opts?.paperTitle,
    title: kind === "paper" ? (opts?.paperTitle ?? "Paper chat") : "New chat",
    turns: [],
    createdAt: now,
    updatedAt: now,
  };
}

export function saveSession(session: ChatSession): ChatSession {
  const updated: ChatSession = { ...session, updatedAt: Date.now() };
  const all = readAll();
  const idx = all.findIndex((s) => s.id === session.id);
  if (idx >= 0) all[idx] = updated;
  else all.push(updated);
  writeAll(all);
  return updated;
}

export function deleteSession(id: string): void {
  writeAll(readAll().filter((s) => s.id !== id));
}

/** Derives a display title from the first user message, unless this is a paper
 * session (which already has a stable title from the paper itself). */
export function deriveTitle(session: ChatSession, turns: ChatTurn[]): string {
  if (session.kind === "paper") return session.title;
  const firstUser = turns.find((t) => t.role === "user");
  if (!firstUser) return "New chat";
  const text = firstUser.content.trim();
  return text.length > TITLE_MAX_CHARS ? `${text.slice(0, TITLE_MAX_CHARS)}…` : text;
}

/** Every paper id seen across all stored sessions, most recently used first.
 *
 * This is what makes a paper picker possible without a server-side "recently viewed"
 * table: the sessions already record which papers each answer cited, so the set of
 * papers the user has actually encountered is derivable from what is on disk. */
export function recentPaperIds(limit = 40): string[] {
  const seen = new Set<string>();
  const ids: string[] = [];
  for (const session of listSessions()) {
    if (session.paperId && !seen.has(session.paperId)) {
      seen.add(session.paperId);
      ids.push(session.paperId);
    }
    for (const turn of session.turns) {
      for (const source of turn.sources ?? []) {
        // Deduped by canonical id so a paper stored under both a bare and a
        // versioned id does not appear twice in a list the user is picking from.
        const key = source.paper_id.replace(/v\d+$/, "");
        if (seen.has(key)) continue;
        seen.add(key);
        ids.push(source.paper_id);
        if (ids.length >= limit) return ids;
      }
    }
  }
  return ids;
}
