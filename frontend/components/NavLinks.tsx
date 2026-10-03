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
    <div className="flex gap-1 text-sm">
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
            className={`rounded-full px-3 py-1.5 transition-colors ${
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
