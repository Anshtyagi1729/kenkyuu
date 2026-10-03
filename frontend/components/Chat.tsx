"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import type { ChatTurn, Message, Source, StreamEvent } from "@/lib/api";
import MarkdownAnswer from "@/components/MarkdownAnswer";
import AgentActivity from "@/components/AgentActivity";
import { fetchPaper, peekPaper } from "@/lib/paperCache";

interface ChatProps {
  turns: ChatTurn[];
  onTurnsChange: (turns: ChatTurn[]) => void;
  onSend: (history: Message[], onEvent: (event: StreamEvent) => void, signal: AbortSignal) => Promise<void>;
  placeholder?: string;
  /** Rendered when there are no turns yet. Receives a `send` callback so an empty
   * state can offer example prompts that submit directly, rather than asking the
   * user to retype them. */
  emptyState?: React.ReactNode | ((send: (prompt: string) => void) => React.ReactNode);
}

export default function Chat({ turns, onTurnsChange, onSend, placeholder, emptyState }: ChatProps) {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // The in-progress assistant reply, kept as local render state (not pushed through
  // onTurnsChange) while streaming - onTurnsChange persists to localStorage on every
  // call, and doing that on every token would mean stringifying + writing the whole
  // session dozens of times a second.
  const [streamingText, setStreamingText] = useState<string | null>(null);
  // What the agent is doing, accumulated in order. Cleared per request, and kept
  // visible until the first answer token arrives.
  const [activity, setActivity] = useState<string[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, loading, streamingText]);

  // Abort any in-flight request if the component unmounts mid-request (e.g. user
  // navigates away) - avoids a dangling fetch and a state update after unmount.
  useEffect(() => {
    return () => controllerRef.current?.abort();
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    await send(input);
  }

  async function send(raw: string) {
    const question = raw.trim();
    if (!question || loading) return;

    const nextTurns: ChatTurn[] = [...turns, { role: "user", content: question }];
    onTurnsChange(nextTurns);
    setInput("");
    setLoading(true);
    setError(null);
    setStreamingText("");
    setActivity([]);

    const controller = new AbortController();
    controllerRef.current = controller;

    let answerText = "";
    let sources: Source[] = [];
    // Mirrored locally as well as in state: the turn is persisted in the same tick the
    // stream finishes, and React state updates are not visible synchronously, so
    // reading `activity` there would save an empty list.
    const steps: string[] = [];

    try {
      const history: Message[] = nextTurns.map((t) => ({ role: t.role, content: t.content }));
      await onSend(
        history,
        (event) => {
          if (event.type === "activity") {
            steps.push(event.text);
            setActivity((prev) => [...prev, event.text]);
          } else if (event.type === "token") {
            answerText += event.text;
            setStreamingText(answerText);
          } else if (event.type === "done") {
            sources = event.sources;
          } else if (event.type === "error") {
            setError(event.message);
          }
        },
        controller.signal,
      );
      if (answerText)
        onTurnsChange([...nextTurns, { role: "assistant", content: answerText, sources, steps }]);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        // User hit Stop - not a real error. Keep whatever streamed in so far rather
        // than discarding it, same as leaving their message with no reply used to.
        if (answerText)
          onTurnsChange([...nextTurns, { role: "assistant", content: answerText, sources, steps }]);
      } else {
        setError(err instanceof Error ? err.message : "Something went wrong.");
      }
    } finally {
      setLoading(false);
      setStreamingText(null);
      setActivity([]);
      controllerRef.current = null;
    }
  }

  function handleStop() {
    controllerRef.current?.abort();
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="scroll-area flex-1 overflow-y-auto pr-3">
        {turns.length === 0 && (typeof emptyState === "function" ? emptyState(send) : emptyState)}
        <div className="flex flex-col gap-5 pb-4">
          {turns.map((turn, i) => (
            <div key={i} className={turn.role === "user" ? "flex justify-end" : "flex justify-start"}>
              {turn.role === "user" ? (
                <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-accent px-4 py-2.5 text-[15px] text-white">
                  {turn.content}
                </div>
              ) : (
                <div className="flex max-w-[85%] min-w-0 flex-col gap-3">
                  {turn.steps && turn.steps.length > 0 && <StepsDisclosure steps={turn.steps} />}
                  <div className="rounded-2xl rounded-bl-sm border border-border bg-surface px-4 py-3">
                    <MarkdownAnswer text={turn.content} />
                  </div>
                  {turn.sources && turn.sources.length > 0 && <SourceList sources={turn.sources} />}
                </div>
              )}
            </div>
          ))}
          {streamingText !== null && (
            <div className="flex justify-start">
              <div className="max-w-[85%] rounded-2xl rounded-bl-sm border border-border bg-surface px-4 py-3">
                {streamingText ? (
                  <MarkdownAnswer text={streamingText} />
                ) : activity.length > 0 ? (
                  <AgentActivity steps={activity} running />
                ) : (
                  <div className="flex items-center gap-1.5">
                    <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-foreground-muted [animation-delay:-0.3s]" />
                    <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-foreground-muted [animation-delay:-0.15s]" />
                    <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-foreground-muted" />
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
        <div ref={bottomRef} />
      </div>

      {error && <p className="pb-2 text-sm text-red-600">{error}</p>}

      <form onSubmit={handleSubmit} className="flex gap-2 border-t border-border pt-4">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={placeholder}
          className="flex-1 rounded-full border border-border bg-surface px-4 py-2.5 text-[15px] outline-none transition-shadow focus:border-accent focus:ring-4 focus:ring-accent/10"
        />
        {loading ? (
          <button
            type="button"
            onClick={handleStop}
            className="rounded-full border border-accent px-5 py-2.5 text-sm font-medium text-accent transition-colors hover:bg-accent hover:text-white"
          >
            Stop
          </button>
        ) : (
          <button
            type="submit"
            disabled={!input.trim()}
            className="rounded-full bg-accent px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-accent-hover disabled:opacity-40"
          >
            Send
          </button>
        )}
      </form>
    </div>
  );
}

/** Collapses a source's cited page numbers into a compact range string, e.g.
 * sources on pages 1, 2, 3, 5, 6, 8 -> "pp. 1-3, 5-6, 8". Null if no page data
 * (e.g. the source is only the abstract, not full text). */
function formatPages(sources: Source[]): string | null {
  const pages = new Set<number>();
  for (const s of sources) {
    if (s.start_page == null || s.end_page == null) continue;
    for (let p = s.start_page; p <= s.end_page; p++) pages.add(p);
  }
  if (pages.size === 0) return null;

  const sorted = [...pages].sort((a, b) => a - b);
  const ranges: string[] = [];
  let start = sorted[0];
  let prev = sorted[0];
  for (let i = 1; i <= sorted.length; i++) {
    const cur = sorted[i];
    if (cur !== prev + 1) {
      ranges.push(start === prev ? `${start}` : `${start}-${prev}`);
      start = cur;
    }
    prev = cur;
  }
  return `p${pages.size > 1 ? "p" : ""}. ${ranges.join(", ")}`;
}

function SourceList({ sources }: { sources: Source[] }) {
  const byPaper = new Map<string, Source[]>();
  for (const source of sources) {
    if (!byPaper.has(source.paper_id)) byPaper.set(source.paper_id, []);
    byPaper.get(source.paper_id)!.push(source);
  }

  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-xs font-medium tracking-wide text-foreground-muted uppercase">Sources</span>
      <div className="flex flex-wrap gap-2">
        {[...byPaper.entries()].map(([paperId, chunks]) => {
          const pages = formatPages(chunks);
          return (
            <Link
              key={paperId}
              href={`/paper/${encodeURIComponent(paperId)}`}
              title={paperId}
              className="flex max-w-[22rem] min-w-0 items-center rounded-full border border-border bg-surface px-3 py-1 text-xs font-medium text-foreground-muted transition-colors hover:border-accent hover:text-accent"
            >
              <SourceLabel paperId={paperId} />
              {pages && <span className="shrink-0 text-foreground-muted/60"> · {pages}</span>}
            </Link>
          );
        })}
      </div>
    </div>
  );
}


/** The agent's steps for a finished answer, collapsed by default.
 *
 * Collapsed because once the answer exists the answer is the point; expandable
 * because "why did it tell me this" is a fair question, and for this system the
 * honest response is the list of things it actually did. */
function StepsDisclosure({ steps }: { steps: string[] }) {
  return (
    <details className="group">
      <summary className="flex w-fit cursor-pointer list-none items-center gap-1 text-xs text-foreground-muted hover:text-accent [&::-webkit-details-marker]:hidden">
        {steps.length} step{steps.length === 1 ? "" : "s"}
        <span className="transition-transform group-open:rotate-90">›</span>
      </summary>
      <div className="mt-2 border-l-2 border-border pl-3">
        <AgentActivity steps={steps} />
      </div>
    </details>
  );
}


/** A source pill's text: the paper's title once known, its id until then.
 *
 * The pills previously read "1810.04805v2". That is the correct identifier and a
 * useless label - it asks the reader to have memorised arXiv ids to know what the
 * answer was based on, which defeats the purpose of showing sources at all. The id
 * stays in the tooltip, where it is useful for citing.
 *
 * Falling back to the id rather than a spinner keeps the pill a stable width and
 * always meaningful, so a slow lookup degrades to the old behaviour instead of a
 * flicker. */
function SourceLabel({ paperId }: { paperId: string }) {
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

  return <span className="truncate">{title ?? paperId}</span>;
}
