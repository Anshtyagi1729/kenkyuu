"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import cytoscape, { type Core, type ElementDefinition, type NodeSingular } from "cytoscape";
import { getCitationGraph, type CitationGraphData, type GraphEdge, type GraphNode } from "@/lib/api";
import { formatCount } from "@/lib/formatMetrics";

/** A paper's citation neighbourhood, laid out as a timeline.
 *
 * The earlier version drew every neighbour as an identical blank circle in a starburst
 * around the centre. It said nothing at a glance: references and citing papers were
 * told apart only by tiny arrowheads, a landmark looked the same as a paper nobody
 * reads, and every title was hidden until hovered. It also auto-zoomed to fit, so a
 * paper with one neighbour filled the screen with two giant circles.
 *
 * Each visual channel now carries one fact:
 *   x position   publication year - what a paper builds on sits to its left, what
 *                builds on it to its right, so the picture reads as a timeline
 *   colour       relation - references, citing papers, the paper itself
 *   size         citation count (log) - the influential neighbours stand out
 *   label        shown for the paper itself and the most-cited neighbours; others on hover
 *
 * The side panel lists the same papers by citation count. It is the readable form of the
 * graph, it works when the graph is tiny, and clicking a title highlights its node. */

type Relation = "center" | "reference" | "citation" | "expanded";

interface Item extends Omit<GraphNode, "relation"> {
  relation: Relation;
}

interface GraphState {
  /** The centre paper this state belongs to, so a stale response is never shown. */
  forId: string;
  items: Item[];
  edges: GraphEdge[];
  source: string | null;
  totalReferences: number | null;
  totalCitations: number | null;
  /** Papers whose own links have already been pulled in. State rather than a ref,
   * because the Expand button's label is rendered from it. */
  expanded: string[];
}

// Cytoscape draws to a canvas and cannot read CSS variables; these mirror globals.css
// and the README diagram palette so the graph matches the rest of the interface.
const COLORS: Record<Relation, { fill: string; border: string; edge: string }> = {
  center: { fill: "#c96442", border: "#a8512f", edge: "#c96442" },
  reference: { fill: "#dde5ef", border: "#5a7799", edge: "#b9c7d9" },
  citation: { fill: "#d6ebd8", border: "#4a8a52", edge: "#b3d3b8" },
  expanded: { fill: "#eeebe4", border: "#a39c8e", edge: "#d9d3c7" },
};
const FOREGROUND = "#262521";
const MUTED = "#73695c";
const CANVAS_BG = "#faf9f5";
const CANVAS_FONT = "ui-sans-serif, system-ui, sans-serif";

const YEAR_SPACING = 104; // px between adjacent year columns
const ROW_SPACING = 56; // px between nodes stacked in one column
const MAX_ROWS = 8; // nodes per column before spilling into a neighbouring sub-column
const SUB_COLUMN_SPACING = 48; // wider than the largest node, so sub-columns never overlap
const MAX_COLUMNS = 9; // year columns; beyond this, neighbouring years share a column
const ALWAYS_LABELLED = 3; // most-cited neighbours per relation that keep a visible label
const MAX_FIT_ZOOM = 1.25; // stops a tiny graph from being blown up to fill the canvas

const RELATION_LABEL: Record<Relation, string> = {
  center: "This paper",
  reference: "Cited by this paper",
  citation: "Cites this paper",
  expanded: "Linked through an expanded paper",
};

function nodeSize(citations: number | null | undefined, relation: Relation): number {
  // log10 because citation counts span six orders of magnitude: 0 -> 12px, 100 -> 26px,
  // 10,000 -> 40px. Linear sizing would make one landmark the only visible node.
  const size = Math.min(46, 12 + 7 * Math.log10(1 + (citations ?? 0)));
  return relation === "center" ? Math.max(size, 30) : size;
}

function shortLabel(title: string, max = 34): string {
  return title.length > max ? `${title.slice(0, max - 1)}…` : title;
}

function toItems(data: CitationGraphData): Item[] {
  return data.nodes.map((n) => ({ ...n, relation: (n.relation ?? "expanded") as Relation }));
}

/** Splits sorted distinct years into at most `count` contiguous groups. */
function binYears(years: number[], count: number): number[][] {
  if (count <= 0 || years.length === 0) return [];
  if (years.length <= count) return years.map((y) => [y]);
  const bins: number[][] = [];
  for (let i = 0; i < count; i++) {
    const from = Math.floor((i * years.length) / count);
    const to = Math.floor(((i + 1) * years.length) / count);
    bins.push(years.slice(from, to));
  }
  return bins;
}

/** Timeline positions. Columns run oldest to newest; the paper's own year always has a
 * column, and the years before and after it share the remaining columns.
 *
 * Capped at MAX_COLUMNS because one column per year produced a strip about 2,200px wide
 * and three rows tall for the Transformer paper (references back to 1997, citations to
 * 2024). Fitted into the panel, every label and node shrank below legibility. Grouping
 * neighbouring years keeps the picture roughly as tall as it is wide, at a readable
 * zoom; a grouped column is labelled with its range ("1997-2013"). */
function computeLayout(items: Item[]) {
  const center = items.find((i) => i.relation === "center");
  const knownYears = items.map((i) => i.year).filter((y): y is number => typeof y === "number");
  const centerYear =
    center?.year ?? (knownYears.length ? [...knownYears].sort((a, b) => a - b)[Math.floor(knownYears.length / 2)] : 0);

  const distinct = [...new Set(knownYears)].sort((a, b) => a - b);
  const before = distinct.filter((y) => y < centerYear);
  const after = distinct.filter((y) => y > centerYear);
  const sideColumns = MAX_COLUMNS - 1;
  // Share the columns between the two sides in proportion to how many years each spans,
  // giving any non-empty side at least one.
  let leftCount = before.length && after.length
    ? Math.max(1, Math.round((sideColumns * before.length) / (before.length + after.length)))
    : before.length ? sideColumns : 0;
  leftCount = Math.min(leftCount, before.length, sideColumns - (after.length ? 1 : 0));
  const rightCount = Math.min(after.length, sideColumns - leftCount);
  // The paper itself always has a column of its own. Same-year neighbours go in a
  // column just before it (papers it cites) or just after it (papers citing it): sharing
  // its column put their labels underneath its three-line title, where they collided.
  const sameYear = items.filter((i) => i.relation !== "center" && (i.year ?? centerYear) === centerYear);
  const sameYearLeft = sameYear.some((i) => i.relation !== "citation");
  const sameYearRight = sameYear.some((i) => i.relation === "citation");

  type Column = { years: number[]; label: string; side?: "left" | "right" | "center" };
  const toColumn = (bin: number[]): Column => ({
    years: bin,
    label: bin.length > 1 ? `${bin[0]}–${String(bin[bin.length - 1]).slice(2)}` : String(bin[0]),
  });
  const columnsSpec: Column[] = [
    ...binYears(before, leftCount).map(toColumn),
    ...(sameYearLeft ? [{ years: [centerYear], label: "", side: "left" as const }] : []),
    { years: [centerYear], label: String(centerYear), side: "center" },
    ...(sameYearRight ? [{ years: [centerYear], label: "", side: "right" as const }] : []),
    ...binYears(after, rightCount).map(toColumn),
  ];

  const columnOf = (item: Item) => {
    const y = item.year ?? centerYear;
    if (y === centerYear) {
      const side = item.relation === "center" ? "center" : item.relation === "citation" ? "right" : "left";
      return columnsSpec.findIndex((c) => c.side === side);
    }
    const index = columnsSpec.findIndex((c) => !c.side && c.years.includes(y));
    return index >= 0 ? index : columnsSpec.findIndex((c) => c.side === "center");
  };

  const columns = new Map<number, Item[]>();
  for (const item of items) {
    const col = columnOf(item);
    columns.set(col, [...(columns.get(col) ?? []), item]);
  }

  const positions: Record<string, { x: number; y: number }> = {};
  let lowest = 0;
  for (const [col, column] of columns) {
    column.sort((a, b) => {
      if (a.relation === "center") return -1;
      if (b.relation === "center") return 1;
      return (b.citation_count ?? 0) - (a.citation_count ?? 0);
    });
    const subColumns = Math.ceil(column.length / MAX_ROWS);
    const isCenterColumn = column[0]?.relation === "center";
    // Only the paper itself sits on the centre line. Other columns start half a row off
    // it, alternating up and down from one column to the next. Otherwise the most-cited
    // paper of every column - exactly the ones that keep a permanent label - lands on
    // the same row, and their labels run into each other.
    const offset = isCenterColumn ? 0 : col % 2 === 0 ? 0.5 : -0.5;
    column.forEach((item, i) => {
      const slot = i % MAX_ROWS;
      const sub = Math.floor(i / MAX_ROWS);
      const step = Math.ceil(slot / 2) * (slot % 2 === 1 ? -1 : 1); // 0, -1, 1, -2, 2, ...
      const y = (step + offset) * ROW_SPACING;
      positions[item.paper_id] = {
        x: col * YEAR_SPACING + (sub - (subColumns - 1) / 2) * SUB_COLUMN_SPACING,
        y,
      };
      lowest = Math.max(lowest, y);
    });
  }

  const axisY = lowest + 64;
  const axis = columnsSpec
    .map((column, col) => ({ key: `${col}`, label: column.label, x: col * YEAR_SPACING, y: axisY }))
    .filter((a) => a.label);
  return { positions, axis, hasYears: knownYears.length > 0 };
}

function buildElements(items: Item[], edges: GraphEdge[]): ElementDefinition[] {
  const { positions, axis, hasYears } = computeLayout(items);

  // The most-cited neighbours of each kind keep a permanent label; the rest show theirs
  // on hover. Labelling all forty at once is an unreadable wall of overlapping text.
  const labelled = new Set<string>();
  for (const relation of ["reference", "citation", "expanded"] as Relation[]) {
    items
      .filter((i) => i.relation === relation)
      .sort((a, b) => (b.citation_count ?? 0) - (a.citation_count ?? 0))
      .slice(0, ALWAYS_LABELLED)
      .forEach((i) => labelled.add(i.paper_id));
  }

  const nodes: ElementDefinition[] = items.map((item) => ({
    group: "nodes",
    data: {
      id: item.paper_id,
      label: item.relation === "center" ? shortLabel(item.title, 48) : shortLabel(item.title, 30),
      size: nodeSize(item.citation_count, item.relation),
    },
    position: positions[item.paper_id],
    classes: [item.relation, labelled.has(item.paper_id) ? "labelled" : ""].join(" ").trim(),
  }));

  const axisNodes: ElementDefinition[] = hasYears
    ? axis.map((a) => ({
        group: "nodes",
        data: { id: `axis-${a.key}`, label: a.label },
        position: { x: a.x, y: a.y },
        classes: "axis",
        selectable: false,
        grabbable: false,
      }))
    : [];

  const ids = new Set(items.map((i) => i.paper_id));
  const relationOf = new Map(items.map((i) => [i.paper_id, i.relation]));
  const edgeEls: ElementDefinition[] = edges
    .filter((e) => ids.has(e.source) && ids.has(e.target))
    .map((e) => {
      // An edge takes the colour of its non-centre end, so the lines into the centre
      // read as the same two groups the nodes do.
      const ends = [relationOf.get(e.source), relationOf.get(e.target)];
      const relation = (ends.find((r) => r && r !== "center") ?? "expanded") as Relation;
      return {
        group: "edges",
        data: { id: `${e.source}->${e.target}`, source: e.source, target: e.target },
        classes: `edge-${relation}`,
      };
    });

  return [...axisNodes, ...nodes, ...edgeEls];
}

function fitCapped(cy: Core) {
  cy.fit(undefined, 48);
  if (cy.zoom() > MAX_FIT_ZOOM) {
    cy.zoom(MAX_FIT_ZOOM);
    cy.center();
  }
}

export default function CitationGraph({ centerPaperId }: { centerPaperId: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const [graph, setGraph] = useState<GraphState | null>(null);
  const [failure, setFailure] = useState<{ forId: string; message: string } | null>(null);
  const [selection, setSelection] = useState<{ forId: string; paperId: string } | null>(null);
  const [expanding, setExpanding] = useState<string | null>(null);

  const current = graph?.forId === centerPaperId ? graph : null;
  const error = failure?.forId === centerPaperId ? failure.message : null;
  const loading = !current && !error;
  const selectedId = selection?.forId === centerPaperId ? selection.paperId : null;
  const selected = current?.items.find((i) => i.paper_id === selectedId) ?? null;

  // The tap handler is registered once, so it reads the current centre through a ref.
  const centerPaperIdRef = useRef(centerPaperId);
  useEffect(() => {
    centerPaperIdRef.current = centerPaperId;
  }, [centerPaperId]);

  // Mount Cytoscape once; elements are replaced imperatively when the data changes.
  useEffect(() => {
    if (!containerRef.current) return;
    const cy = cytoscape({
      container: containerRef.current,
      minZoom: 0.25,
      maxZoom: 2.5,
      wheelSensitivity: 0.3,
      boxSelectionEnabled: false,
      style: [
        {
          selector: "node",
          style: {
            width: "data(size)",
            height: "data(size)",
            "border-width": 1.5,
            "background-color": COLORS.expanded.fill,
            "border-color": COLORS.expanded.border,
          },
        },
        ...(["center", "reference", "citation", "expanded"] as Relation[]).map((r) => ({
          selector: `node.${r}`,
          style: { "background-color": COLORS[r].fill, "border-color": COLORS[r].border },
        })),
        {
          selector: "node.labelled, node.center, node.hovered, node:selected",
          style: {
            label: "data(label)",
            "font-family": CANVAS_FONT,
            "font-size": 10,
            color: FOREGROUND,
            "text-valign": "bottom",
            "text-margin-y": 4,
            "text-wrap": "wrap",
            "text-max-width": "88px",
            "text-background-color": CANVAS_BG,
            "text-background-opacity": 0.92,
            "text-background-padding": "2px",
            "text-background-shape": "roundrectangle",
          },
        },
        {
          selector: "node.center",
          style: { "font-size": 11, "font-weight": "bold", "border-width": 3, "text-max-width": "140px" },
        },
        {
          selector: "node:selected",
          style: { "border-width": 3, "border-color": FOREGROUND, "z-index": 10 },
        },
        {
          selector: "node.axis",
          style: {
            width: 1,
            height: 1,
            "background-opacity": 0,
            "border-width": 0,
            label: "data(label)",
            "font-family": CANVAS_FONT,
            "font-size": 10,
            color: MUTED,
            "text-valign": "center",
            events: "no",
          },
        },
        {
          selector: "edge",
          style: {
            width: 1,
            "curve-style": "straight",
            opacity: 0.7,
            "target-arrow-shape": "triangle",
            "arrow-scale": 0.55,
            "line-color": COLORS.expanded.edge,
            "target-arrow-color": COLORS.expanded.edge,
          },
        },
        ...(["reference", "citation", "expanded"] as Relation[]).map((r) => ({
          selector: `edge.edge-${r}`,
          style: { "line-color": COLORS[r].edge, "target-arrow-color": COLORS[r].edge },
        })),
        // Hovering a paper dims everything outside its own neighbourhood, so its links
        // can be followed even in a dense graph.
        { selector: ".faded", style: { opacity: 0.12 } },
        { selector: "edge.lit", style: { opacity: 1, width: 1.8 } },
      ],
    });
    cyRef.current = cy;

    const focus = (node: NodeSingular) => {
      const hood = node.closedNeighborhood();
      cy.elements().not(hood).not(".axis").addClass("faded");
      hood.edges().addClass("lit");
      node.addClass("hovered");
    };
    const unfocus = () => cy.elements().removeClass("faded lit hovered");

    cy.on("mouseover", "node", (evt) => {
      if (!evt.target.hasClass("axis")) focus(evt.target);
    });
    cy.on("mouseout", "node", unfocus);
    cy.on("tap", "node", (evt) => {
      const node = evt.target as NodeSingular;
      if (node.hasClass("axis")) return;
      cy.elements(":selected").unselect();
      node.select();
      setSelection({ forId: centerPaperIdRef.current, paperId: node.id() });
    });
    cy.on("tap", (evt) => {
      if (evt.target === cy) {
        cy.elements(":selected").unselect();
        setSelection(null);
      }
    });

    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, []);

  // Fetch the neighbourhood whenever the centre paper changes. State is stored with the
  // id it belongs to, so nothing has to be reset synchronously and a slow response for a
  // previous paper can never overwrite the current one.
  useEffect(() => {
    let cancelled = false;
    getCitationGraph(centerPaperId)
      .then((data) => {
        if (cancelled) return;
        setGraph({
          forId: centerPaperId,
          items: toItems(data),
          edges: data.edges,
          source: data.source ?? null,
          totalReferences: data.total_references ?? null,
          totalCitations: data.total_citations ?? null,
          expanded: [],
        });
      })
      .catch((err) => {
        if (!cancelled) {
          setFailure({ forId: centerPaperId, message: err instanceof Error ? err.message : "Could not load the citation graph." });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [centerPaperId]);

  // Draw whenever the data changes. The layout is deterministic (year and citation
  // order), so redrawing after an expansion places every paper where it belongs rather
  // than wherever a physics simulation happened to settle.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !current) return;
    cy.elements().remove();
    cy.add(buildElements(current.items, current.edges));
    fitCapped(cy);
  }, [current]);

  // Keep the canvas selection in step with selections made from the side panel.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.elements(":selected").unselect();
    if (selectedId) cy.getElementById(selectedId).select();
  }, [selectedId, current]);

  function selectFromList(paperId: string) {
    setSelection({ forId: centerPaperId, paperId });
    const node = cyRef.current?.getElementById(paperId);
    if (node && node.nonempty()) cyRef.current?.center(node);
  }

  function hoverFromList(paperId: string | null) {
    const cy = cyRef.current;
    if (!cy) return;
    cy.elements().removeClass("faded lit hovered");
    if (!paperId) return;
    const node = cy.getElementById(paperId);
    if (node.empty()) return;
    const hood = node.closedNeighborhood();
    cy.elements().not(hood).not(".axis").addClass("faded");
    hood.edges().addClass("lit");
    node.addClass("hovered");
  }

  async function handleExpand(paperId: string) {
    if (!current || current.expanded.includes(paperId)) return;
    setExpanding(paperId);
    try {
      const data = await getCitationGraph(paperId);
      setGraph((prev) => {
        if (!prev || prev.forId !== centerPaperId) return prev;
        const known = new Set(prev.items.map((i) => i.paper_id));
        // Papers reached through another paper are a different kind of neighbour from
        // the centre's own references and citations, and get their own neutral colour.
        const added: Item[] = data.nodes
          .filter((n) => !known.has(n.paper_id))
          .map((n) => ({ ...n, relation: "expanded" }));
        const edgeKeys = new Set(prev.edges.map((e) => `${e.source}->${e.target}`));
        const newEdges = data.edges.filter((e) => !edgeKeys.has(`${e.source}->${e.target}`));
        return {
          ...prev,
          items: [...prev.items, ...added],
          edges: [...prev.edges, ...newEdges],
          expanded: [...prev.expanded, paperId],
        };
      });
    } catch {
      // Not marked as expanded, so the button stays available for a retry.
    } finally {
      setExpanding(null);
    }
  }

  const groups = useMemo(() => {
    const byCitations = (a: Item, b: Item) => (b.citation_count ?? 0) - (a.citation_count ?? 0);
    const items = current?.items ?? [];
    return {
      references: items.filter((i) => i.relation === "reference").sort(byCitations),
      citations: items.filter((i) => i.relation === "citation").sort(byCitations),
      expanded: items.filter((i) => i.relation === "expanded").sort(byCitations),
    };
  }, [current]);

  const isolated = current !== null && current.items.length <= 1;
  const isExpanded = (paperId: string) => current?.expanded.includes(paperId) ?? false;

  return (
    <div className="grid min-h-0 flex-1 gap-4 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <div className="relative flex min-h-[520px] flex-col overflow-hidden rounded-2xl border border-border bg-background">
        <Legend />
        {current && !isolated && (
          <button
            onClick={() => cyRef.current && fitCapped(cyRef.current)}
            className="absolute top-3 right-3 z-10 rounded-full border border-border bg-surface px-3 py-1 text-xs font-medium text-foreground-muted transition-colors hover:border-accent hover:text-accent"
          >
            Fit
          </button>
        )}
        {loading && (
          <div className="absolute inset-0 z-20 flex items-center justify-center bg-background/80 text-sm text-foreground-muted">
            Loading citation graph…
          </div>
        )}
        {error && (
          <div className="absolute inset-0 z-20 flex items-center justify-center bg-background/95 px-6 text-center text-sm text-red-600">
            {error}
          </div>
        )}
        {isolated && (
          <div className="absolute inset-0 z-20 flex items-center justify-center px-8 text-center text-sm text-foreground-muted">
            No citation links were found for this paper in OpenAlex or Semantic Scholar.
          </div>
        )}
        <div ref={containerRef} className="min-h-0 flex-1" />
        {current && !isolated && (
          <p className="pointer-events-none absolute bottom-2 left-3 text-[11px] text-foreground-muted">
            Older ← publication year → newer · size = citations · hover to trace links
          </p>
        )}
      </div>

      <aside className="flex min-h-0 flex-col gap-3">
        {selected && (
          <div className="flex flex-col gap-2 rounded-xl border border-border bg-surface p-4">
            <span className="text-[11px] font-medium tracking-wide text-foreground-muted uppercase">
              {RELATION_LABEL[selected.relation]}
            </span>
            <p className="text-sm leading-snug font-medium text-foreground">{selected.title}</p>
            <p className="text-xs text-foreground-muted">
              {[selected.year, selected.citation_count != null ? `${formatCount(selected.citation_count)} citations` : null]
                .filter(Boolean)
                .join(" · ")}
            </p>
            <div className="mt-1 flex gap-2">
              <Link
                href={`/paper/${encodeURIComponent(selected.paper_id)}`}
                className="rounded-full border border-border px-3 py-1.5 text-xs font-medium text-foreground-muted transition-colors hover:border-accent hover:text-accent"
              >
                Open paper
              </Link>
              {selected.relation !== "center" && (
                <button
                  onClick={() => handleExpand(selected.paper_id)}
                  disabled={expanding !== null || isExpanded(selected.paper_id)}
                  className="rounded-full bg-accent px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-accent-hover disabled:opacity-40"
                >
                  {isExpanded(selected.paper_id)
                    ? "Expanded"
                    : expanding === selected.paper_id
                      ? "Expanding…"
                      : "Expand its links"}
                </button>
              )}
            </div>
          </div>
        )}

        <div className="scroll-area flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto rounded-xl border border-border bg-surface p-4 lg:max-h-[640px]">
          <PaperList
            title="Builds on"
            note={countNote(groups.references.length, current?.totalReferences)}
            relation="reference"
            items={groups.references}
            selectedId={selectedId}
            onSelect={selectFromList}
            onHover={hoverFromList}
          />
          <PaperList
            title="Cited by"
            note={countNote(groups.citations.length, current?.totalCitations)}
            relation="citation"
            items={groups.citations}
            selectedId={selectedId}
            onSelect={selectFromList}
            onHover={hoverFromList}
          />
          {groups.expanded.length > 0 && (
            <PaperList
              title="Reached by expanding"
              note={`${groups.expanded.length}`}
              relation="expanded"
              items={groups.expanded}
              selectedId={selectedId}
              onSelect={selectFromList}
              onHover={hoverFromList}
            />
          )}
          {current?.source && (
            <p className="text-[11px] text-foreground-muted">
              Source: {current.source === "openalex" ? "OpenAlex" : "Semantic Scholar"}. The most-cited neighbours
              are shown first.
            </p>
          )}
        </div>
      </aside>
    </div>
  );
}

function countNote(shown: number, total: number | null | undefined): string {
  if (total != null && total > shown) return `${shown} of ${formatCount(total)}`;
  return `${shown}`;
}

function Legend() {
  const entries: [Relation, string][] = [
    ["center", "This paper"],
    ["reference", "Builds on"],
    ["citation", "Cited by"],
  ];
  return (
    <div className="pointer-events-none absolute top-3 left-3 z-10 flex flex-wrap gap-x-3 gap-y-1 rounded-lg bg-surface/90 px-2.5 py-1.5 text-[11px] text-foreground-muted">
      {entries.map(([relation, label]) => (
        <span key={relation} className="flex items-center gap-1.5">
          <span
            className="inline-block h-2.5 w-2.5 rounded-full border"
            style={{ backgroundColor: COLORS[relation].fill, borderColor: COLORS[relation].border }}
          />
          {label}
        </span>
      ))}
    </div>
  );
}

function PaperList({
  title,
  note,
  relation,
  items,
  selectedId,
  onSelect,
  onHover,
}: {
  title: string;
  note: string;
  relation: Relation;
  items: Item[];
  selectedId: string | null;
  onSelect: (paperId: string) => void;
  onHover: (paperId: string | null) => void;
}) {
  return (
    <section className="flex flex-col gap-1.5">
      <h3 className="flex items-center gap-2 text-[11px] font-medium tracking-wide text-foreground-muted uppercase">
        <span
          className="inline-block h-2 w-2 rounded-full border"
          style={{ backgroundColor: COLORS[relation].fill, borderColor: COLORS[relation].border }}
        />
        {title}
        <span className="font-normal normal-case">{note}</span>
      </h3>
      {items.length === 0 ? (
        <p className="text-xs text-foreground-muted">None found.</p>
      ) : (
        <ul className="flex flex-col">
          {items.map((item) => (
            <li key={item.paper_id}>
              <button
                onClick={() => onSelect(item.paper_id)}
                onMouseEnter={() => onHover(item.paper_id)}
                onMouseLeave={() => onHover(null)}
                className={`flex w-full min-w-0 flex-col rounded-lg px-2 py-1.5 text-left transition-colors hover:bg-surface-muted ${
                  item.paper_id === selectedId ? "bg-accent-soft" : ""
                }`}
              >
                <span className="line-clamp-2 text-xs leading-snug text-foreground">{item.title}</span>
                <span className="text-[11px] text-foreground-muted">
                  {[item.year, item.citation_count != null ? `${formatCount(item.citation_count)} cites` : null]
                    .filter(Boolean)
                    .join(" · ")}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
