"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/search", label: "Search" },
  { href: "/compare", label: "Compare" },
  { href: "/library", label: "Library" },
  { href: "/history", label: "History" },
];

/** Primary navigation, with the current section marked.
 *
 * Every link previously rendered identically, so the app gave no indication of where
 * you were - on the paper detail page, which is reached from Search, nothing was
 * highlighted at all. `startsWith` rather than equality is what makes that case work:
 * /paper/... belongs to the Search flow and should keep Search lit. */
export default function NavLinks() {
  const pathname = usePathname() ?? "";

  return (
    // On a phone the brand and four links need about 460px of a 375px screen. The row
    // scrolls sideways instead of clipping "History" off the edge; on wider screens it
    // simply fits.
    <div className="flex min-w-0 gap-0.5 overflow-x-auto text-[13px] [scrollbar-width:none] sm:gap-1 sm:text-sm">
      {LINKS.map(({ href, label }) => {
        const active =
          pathname === href ||
          pathname.startsWith(`${href}/`) ||
          (href === "/search" && pathname.startsWith("/paper"));
        return (
          <Link
            key={href}
            href={href}
            aria-current={active ? "page" : undefined}
            className={`shrink-0 rounded-full px-2 py-1.5 whitespace-nowrap transition-colors sm:px-3 ${
              active
                ? "bg-accent-soft font-medium text-accent-hover"
                : "text-foreground-muted hover:bg-surface-muted hover:text-accent"
            }`}
          >
            {label}
          </Link>
        );
      })}
    </div>
  );
}
