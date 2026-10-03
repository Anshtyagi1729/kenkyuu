"use client";

/** The agent's work, shown while it happens.
 *
 * A real research question takes tens of seconds of tool calls before a single
 * answer token exists. That time was previously three bouncing dots, which tells the
 * professor nothing and makes a working system look hung.
 *
 * Completed steps stay on screen rather than being replaced. Two reasons: the trail
 * is the evidence for the answer that follows - "searched arXiv, read both papers,
 * checked their citation impact" is what makes the result trustworthy - and it is
 * also how a user discovers that author lookup and citation-neighbour search exist at
 * all, having never been told to ask for them. */
export default function AgentActivity({
  steps,
  running = false,
}: {
  steps: string[];
  /** While true the last step is still in progress and pulses. On a finished answer
   * every step is complete, so nothing should animate - a pulsing dot on a static
   * transcript reads as work still happening. */
  running?: boolean;
}) {
  if (steps.length === 0) return null;

  return (
    <ol className="flex flex-col gap-1.5" aria-live="polite">
      {steps.map((step, i) => {
        const done = !running || i < steps.length - 1;
        return (
          <li key={`${i}-${step}`} className="flex items-start gap-2 text-xs leading-snug">
            <span className="mt-[3px] flex h-3 w-3 shrink-0 items-center justify-center">
              {done ? (
                <svg viewBox="0 0 12 12" className="h-3 w-3 text-accent" aria-hidden>
                  <path
                    d="M2.5 6.2l2.2 2.3L9.5 3.7"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              ) : (
                <span className="h-2 w-2 animate-pulse rounded-full bg-accent" />
              )}
            </span>
            <span className={done ? "text-foreground-muted" : "text-foreground"}>{step}</span>
          </li>
        );
      })}
    </ol>
  );
}
