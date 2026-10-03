"use client";

/** Example questions on the empty search page, grouped by what the system can do.
 *
 * These are discovery, not decoration. The agent can look up an author's papers,
 * find a paper's citation neighbours, and report field-normalized impact - and a
 * blank box with a placeholder about retrieval teaches none of that. A professor
 * typing their first question will ask for a topic, get a topic answer, and never
 * learn the rest existed.
 *
 * The capability label above each example is deliberate: the example shows the
 * phrasing that triggers a tool, and the label says what that tool is, so the
 * pattern generalizes to the user's own questions instead of only working verbatim. */

interface Starter {
  capability: string;
  prompt: string;
}

const STARTERS: Starter[] = [
  {
    capability: "Find a specific paper",
    prompt: "show me the original transformer paper",
  },
  {
    capability: "Papers by a researcher",
    prompt: "what has Yoshua Bengio published recently?",
  },
  {
    capability: "Citation-network neighbours",
    prompt: "what should I read alongside 1706.03762?",
  },
  {
    capability: "Compare two papers",
    prompt: "compare the evaluation setups of 1706.03762 and 1810.04805",
  },
  {
    capability: "Measured impact",
    prompt: "how influential is LoRA compared to the papers it builds on?",
  },
  {
    capability: "Survey a topic",
    prompt: "what are the main approaches to parameter-efficient fine-tuning?",
  },
];

export default function StarterPrompts({ onPick }: { onPick: (prompt: string) => void }) {
  return (
    <div className="flex flex-col gap-4 py-8">
      <div className="flex flex-col gap-1">
        <p className="text-sm text-foreground-muted">
          Ask a research question and get an answer grounded in real papers, with citation impact
          attached to every recommendation.
        </p>
        <p className="text-xs text-foreground-muted">Try one of these:</p>
      </div>

      <div className="grid gap-2 sm:grid-cols-2">
        {STARTERS.map((starter) => (
          <button
            key={starter.prompt}
            type="button"
            onClick={() => onPick(starter.prompt)}
            className="group flex min-w-0 flex-col gap-1 rounded-xl border border-border bg-surface p-3 text-left transition-colors hover:border-accent"
          >
            <span className="text-[11px] font-medium tracking-wide text-foreground-muted uppercase">
              {starter.capability}
            </span>
            <span className="text-sm leading-snug text-foreground group-hover:text-accent-hover">
              {starter.prompt}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
