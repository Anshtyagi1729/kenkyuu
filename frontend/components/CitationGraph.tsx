"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import cytoscape, { type Core, type NodeSingular } from "cytoscape";
import { getCitationGraph, type CitationGraphData, type GraphNode } from "@/lib/api";

interface CitationGraphProps {
  centerPaperId: string;
}

// Cytoscape renders to a <canvas>, so its style spec can't read CSS custom
// properties (var(--accent) etc.) - these are the same palette values from
// globals.css, copied as literals for the one thing that can't reference them.
const ACCENT = "#c96442";
const ACCENT_HOVER = "#b2532f";
const SURFACE = "#ffffff";
const BORDER = "#e6e2d8";
const FOREGROUND = "#262521";
const CANVAS_FONT = "ui-sans-serif, system-ui, sans-serif";

export default function CitationGraph({ centerPaperId }: CitationGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  // Which nodes we've already fetched neighbors for - a plain ref (not state) since
  // it only gates re-fetching and doesn't need to trigger a render on its own.
  const expandedRef = useRef<Set<string>>(new Set());
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [loading, setLoading] = useState(true);
  const [expanding, setExpanding] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Mount the Cytoscape instance once and tear it down on unmount - graph data is
  // merged into it imperatively (mergeGraph below) rather than re-rendered from React
  // state, since a full teardown/rebuild on every fetch would lose pan/zoom position
  // and restart the layout animation for nodes that were already placed.
  useEffect(() => {
    if (!containerRef.current) return;
    const cy = cytoscape({
      container: containerRef.current,
      style: [
        {
          // Bare dot by default, no label - with 15-50+ neighbors on a typical paper,
          // labeling every node at once was an unreadable wall of overlapping text.
          // A title only really matters for the node you're currently looking at.
          selector: "node",
          style: {
            "background-color": SURFACE,
            "border-width": 2,
            "border-color": BORDER,
            width: 14,
            height: 14,
          },
        },
        {
          selector: "node.center, node:selected, node.hovered",
          style: {
            label: "data(label)",
            "font-size": 11,
            "font-family": CANVAS_FONT,
            color: FOREGROUND,
            "text-wrap": "wrap",
            "text-max-width": "110px",
            "text-valign": "bottom",
            "text-margin-y": 6,
            "text-background-color": "#faf9f5",
            "text-background-opacity": 0.85,
            "text-background-padding": "2px",
          },
        },
        {
          selector: "node.center",
          style: {
            "background-color": ACCENT,
            "border-color": ACCENT_HOVER,
            width: 26,
            height: 26,
            "font-weight": "bold",
          },
        },
        {
          selector: "node:selected",
          style: { "border-color": ACCENT, "border-width": 3 },
        },
        {
          selector: "edge",
          style: {
            width: 1.5,
            "line-color": BORDER,
            "target-arrow-color": BORDER,
            "target-arrow-shape": "triangle",
            "arrow-scale": 0.7,
            "curve-style": "bezier",
          },
        },
      ],
      layout: { name: "grid" },
      wheelSensitivity: 0.3,
    });
    cyRef.current = cy;

    cy.on("tap", "node", (evt) => {
      const node = evt.target as NodeSingular;
      // Cytoscape's tap event doesn't select the node on its own - drive selection
      // explicitly so the node:selected style (keeps its label visible after the
      // mouse moves away) actually engages, not just the React side-panel state.
      cy.elements(":selected").unselect();
      node.select();
      setSelected({ paper_id: node.id(), title: node.data("fullTitle"), year: node.data("year") ?? null });
    });
    cy.on("tap", (evt) => {
      if (evt.target === cy) {
        cy.elements(":selected").unselect();
        setSelected(null);
      }
    });
    // Hover reveals a node's title without needing a click for every one - browsing a
    // graph of 20-50 mostly-unlabeled dots would otherwise mean clicking each in turn
    // just to see what it is.
    cy.on("mouseover", "node", (evt) => evt.target.addClass("hovered"));
    cy.on("mouseout", "node", (evt) => evt.target.removeClass("hovered"));

    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, []);

  function mergeGraph(data: CitationGraphData, centerId: string) {
    const cy = cyRef.current;
    if (!cy) return;
    let addedAny = false;
    for (const node of data.nodes) {
      if (cy.getElementById(node.paper_id).nonempty()) continue;
      cy.add({
        data: { id: node.paper_id, label: truncate(node.title, 45), fullTitle: node.title, year: node.year },
        classes: node.paper_id === centerId ? "center" : undefined,
      });
      addedAny = true;
    }
    for (const edge of data.edges) {
      const edgeId = `${edge.source}->${edge.target}`;
      if (cy.getElementById(edgeId).nonempty()) continue;
      if (cy.getElementById(edge.source).empty() || cy.getElementById(edge.target).empty()) continue;
      cy.add({ data: { id: edgeId, source: edge.source, target: edge.target } });
      addedAny = true;
    }
    // Only re-run layout when something actually changed - React 18 dev-mode
    // double-invokes effects, so the same fetch can resolve twice in a row; with
    // nothing new to place, a second layout pass is pure jank, not just pointless work.
    //
    // Not animated: an animated layout ticks via requestAnimationFrame, which browsers
    // simply don't run for a backgrounded/non-visible tab - if the tab isn't in the
    // foreground at the exact moment this runs (easy to hit: open a paper's graph in a
    // new tab and don't switch to it right away), the animation never advances and
    // every node stays stuck at its pre-layout (0,0) position indefinitely.
    if (addedAny) {
      // cose's defaults (tuned for small graphs) pack nodes far too tightly once a
      // paper has 20-50 neighbors - wider spacing keeps individual dots and edges
      // distinguishable instead of collapsing into one dense clump.
      cy.layout({
        name: "cose",
        animate: false,
        padding: 40,
        nodeRepulsion: () => 12000,
        idealEdgeLength: () => 100,
        gravity: 30,
      }).run();
    }
  }

  useEffect(() => {
    expandedRef.current = new Set();
    setSelected(null);
    setLoading(true);
    setError(null);
    getCitationGraph(centerPaperId)
      .then((data) => {
        expandedRef.current.add(centerPaperId);
        mergeGraph(data, centerPaperId);
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load citation graph."))
      .finally(() => setLoading(false));
    // mergeGraph closes over cyRef, which is stable for the component's lifetime.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [centerPaperId]);

  async function handleExpand(paperId: string) {
    if (expandedRef.current.has(paperId)) return;
    expandedRef.current.add(paperId);
    setExpanding(true);
    try {
      const data = await getCitationGraph(paperId);
      mergeGraph(data, paperId);
    } catch {
      expandedRef.current.delete(paperId); // let the user retry rather than silently locking the button
    } finally {
      setExpanding(false);
    }
  }

  return (
    <div className="relative flex min-h-[600px] flex-1 flex-col overflow-hidden rounded-2xl border border-border bg-surface">
      {loading && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-surface/70 text-sm text-foreground-muted">
          Loading citation graph…
        </div>
      )}
      {error && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-surface/90 px-6 text-center text-sm text-red-600">
          {error}
        </div>
      )}
      <div ref={containerRef} className="min-h-0 flex-1" />
      {selected && (
        <div className="absolute right-4 bottom-4 flex w-72 flex-col gap-2 rounded-xl border border-border bg-surface p-4 shadow-lg">
          <p className="text-sm font-medium text-foreground">{selected.title}</p>
          {selected.year && <p className="text-xs text-foreground-muted">{selected.year}</p>}
          <div className="mt-1 flex gap-2">
            <Link
              href={`/paper/${encodeURIComponent(selected.paper_id)}`}
              className="rounded-full border border-border px-3 py-1.5 text-xs font-medium text-foreground-muted transition-colors hover:border-accent hover:text-accent"
            >
              Open paper
            </Link>
            <button
              onClick={() => handleExpand(selected.paper_id)}
              disabled={expanding || expandedRef.current.has(selected.paper_id)}
              className="rounded-full bg-accent px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-accent-hover disabled:opacity-40"
            >
              {expandedRef.current.has(selected.paper_id) ? "Expanded" : expanding ? "Expanding…" : "Expand"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function truncate(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}
