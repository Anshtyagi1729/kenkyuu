import Link from "next/link";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

// Matches both "[2406.16167v1]" and bare "2406.16167v1" - models are asked to always
// bracket citations but don't reliably do so inside markdown tables in practice.
// Every match gets rewritten into a proper markdown link so ReactMarkdown renders it
// through the custom `a` component below, regardless of the original formatting.
const ARXIV_ID_TOKEN = /\[?(\d{4}\.\d{4,5}(?:v\d+)?)\]?/g;

function linkifyCitations(text: string): string {
  return text.replace(ARXIV_ID_TOKEN, (_match, id: string) => `[${id}](/paper/${id})`);
}

// remark-math only recognizes $...$ / $$...$$ delimiters, but text pulled from paper
// PDFs (and models echoing that source formatting) commonly uses LaTeX's own \(...\) /
// \[...\] delimiters instead. Left alone, CommonMark's backslash-escaping silently eats
// the \( \) \[ \] themselves (backslash+punctuation is an escape sequence), so the math
// never even reaches remark-math - it just prints as garbled raw LaTeX. Normalize to
// dollar delimiters before any markdown parsing happens, rather than depend on the model
// choosing a particular delimiter style.
function normalizeLatexDelimiters(text: string): string {
  // Negative lookbehind on both ends: a `\[`/`\]`/`\(`/`\)` only counts as a real
  // delimiter if it's a single backslash, not the tail of a `\\[...]` LaTeX row-break
  // (used inside \begin{aligned}...\end{aligned} for extra spacing, e.g. "\\[4pt]") -
  // otherwise that second backslash+bracket gets misread as an opening math delimiter
  // and swallows everything up to the next `\]` as bogus "math content".
  return text
    .replace(/(?<!\\)\\\[([\s\S]+?)(?<!\\)\\\]/g, (_match, inner: string) => `$$${inner}$$`)
    .replace(/(?<!\\)\\\(([\s\S]+?)(?<!\\)\\\)/g, (_match, inner: string) => `$${inner}$`);
}

const components: Components = {
  a: ({ href, children }) => {
    if (href?.startsWith("/paper/")) {
      return (
        <Link
          href={href}
          className="mx-0.5 rounded-full bg-accent-soft px-1.5 py-0.5 text-[0.85em] font-medium text-accent-hover no-underline hover:bg-accent hover:text-white"
        >
          {children}
        </Link>
      );
    }
    return (
      <a href={href} target="_blank" rel="noreferrer" className="text-accent underline hover:text-accent-hover">
        {children}
      </a>
    );
  },
  p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
  ul: ({ children }) => <ul className="mb-3 list-disc space-y-1 pl-5 last:mb-0">{children}</ul>,
  ol: ({ children }) => <ol className="mb-3 list-decimal space-y-1 pl-5 last:mb-0">{children}</ol>,
  li: ({ children }) => <li>{children}</li>,
  strong: ({ children }) => <strong className="font-semibold text-foreground">{children}</strong>,
  h1: ({ children }) => <h3 className="mt-1 mb-2 font-serif text-lg font-semibold text-foreground">{children}</h3>,
  h2: ({ children }) => <h3 className="mt-1 mb-2 font-serif text-base font-semibold text-foreground">{children}</h3>,
  h3: ({ children }) => <h3 className="mt-1 mb-2 font-serif text-base font-semibold text-foreground">{children}</h3>,
  code: ({ children }) => (
    <code className="rounded bg-surface-muted px-1 py-0.5 font-mono text-[0.85em]">{children}</code>
  ),
  table: ({ children }) => (
    <div className="mb-3 overflow-x-auto last:mb-0">
      <table className="w-full border-collapse text-sm">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="border-b border-border">{children}</thead>,
  th: ({ children }) => (
    <th className="px-2 py-1.5 text-left font-medium text-foreground-muted">{children}</th>
  ),
  td: ({ children }) => <td className="border-t border-border px-2 py-1.5 align-top">{children}</td>,
};

export default function MarkdownAnswer({ text }: { text: string }) {
  return (
    <div className="text-[15px] leading-relaxed">
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]} components={components}>
        {linkifyCitations(normalizeLatexDelimiters(text))}
      </ReactMarkdown>
    </div>
  );
}
