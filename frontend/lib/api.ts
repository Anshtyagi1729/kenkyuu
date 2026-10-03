const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface Message {
  role: "user" | "assistant";
  content: string;
}

export interface Source {
  paper_id: string;
  chunk_id: string;
  text: string;
  start_page: number | null;
  end_page: number | null;
}

export interface ChatTurn extends Message {
  sources?: Source[];
  /** The agent's tool steps for this answer, in order. Persisted with the turn so the
   * reasoning behind an answer is still inspectable after the fact, not only while it
   * is being produced. */
  steps?: string[];
}

export type StreamEvent =
  /** A plain-language description of what the agent is doing right now. Arrives
   * before any answer token, repeatedly, while tools run. */
  | { type: "activity"; text: string }
  | { type: "token"; text: string }
  | { type: "done"; sources: Source[] }
  | { type: "error"; message: string };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Request failed (${res.status}): ${detail}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

function postJSON<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
}

/** Parses one "event: X\ndata: Y" block (blank-line-delimited per the SSE spec). The
 * server always JSON-encodes the data payload (even plain token strings) so a token
 * containing meaningful whitespace or a literal newline survives intact - trimming
 * the raw line here would strip real leading spaces between words. */
function parseSSEEvent(raw: string): StreamEvent | null {
  let eventType = "message";
  let rawData = "";
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) eventType = line.slice(6).trim();
    else if (line.startsWith("data:")) rawData += line.startsWith("data: ") ? line.slice(6) : line.slice(5);
  }
  if (!rawData) return null;
  const data = JSON.parse(rawData);
  if (eventType === "activity") return { type: "activity", text: data as string };
  if (eventType === "token") return { type: "token", text: data as string };
  if (eventType === "done") return { type: "done", sources: data as Source[] };
  if (eventType === "error") return { type: "error", message: data as string };
  return null;
}

async function streamRequest(
  path: string,
  body: unknown,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    const detail = await res.text();
    throw new Error(`Request failed (${res.status}): ${detail}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sepIndex;
    while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, sepIndex);
      buffer = buffer.slice(sepIndex + 2);
      const event = parseSSEEvent(rawEvent);
      if (event) onEvent(event);
    }
  }
}

export function search(history: Message[], onEvent: (event: StreamEvent) => void, signal?: AbortSignal): Promise<void> {
  return streamRequest("/search", { history }, onEvent, signal);
}

export function chatWithPaper(
  paperId: string,
  history: Message[],
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  return streamRequest(`/papers/${encodeURIComponent(paperId)}/chat`, { history }, onEvent, signal);
}

export interface Paper {
  paper_id: string;
  title: string;
  abstract: string | null;
  authors: string[];
  published: string | null;
  external_ids: Record<string, string>;
}

export function getPaper(paperId: string): Promise<Paper> {
  return request<Paper>(`/papers/${encodeURIComponent(paperId)}`);
}

export interface GraphNode {
  paper_id: string;
  title: string;
  year: number | null;
}

export interface GraphEdge {
  source: string;
  target: string;
}

export interface CitationGraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export function getCitationGraph(paperId: string): Promise<CitationGraphData> {
  return request<CitationGraphData>(`/papers/${encodeURIComponent(paperId)}/citation-graph`);
}

export interface CompareResponse {
  comparison: string;
  sources: Source[];
}

export function compare(paperIds: string[], focus?: string): Promise<CompareResponse> {
  return postJSON<CompareResponse>("/compare", { paper_ids: paperIds, focus: focus || null });
}

export interface Folder {
  folder_id: number;
  name: string;
  created_at: string;
  paper_count: number;
}

export function listFolders(): Promise<Folder[]> {
  return request<Folder[]>("/folders");
}

export function createFolder(name: string): Promise<Folder> {
  return postJSON<Folder>("/folders", { name });
}

export function deleteFolder(folderId: number): Promise<void> {
  return request<void>(`/folders/${folderId}`, { method: "DELETE" });
}

export function getFolderPapers(folderId: number): Promise<Paper[]> {
  return request<Paper[]>(`/folders/${folderId}/papers`);
}

export function getPaperFolders(paperId: string): Promise<number[]> {
  return request<number[]>(`/papers/${encodeURIComponent(paperId)}/folders`);
}

export function savePaperToFolder(paperId: string, folderId?: number): Promise<{ folder_id: number }> {
  return postJSON(`/papers/${encodeURIComponent(paperId)}/save`, { folder_id: folderId ?? null });
}

export function unsavePaperFromFolder(paperId: string, folderId: number): Promise<void> {
  return request<void>(`/papers/${encodeURIComponent(paperId)}/save?folder_id=${folderId}`, { method: "DELETE" });
}

/** A paper's scientometric scorecard.
 *
 * Every numeric field is nullable, and null means "not computed" - never zero. The
 * distinction is load-bearing: a 2026 preprint genuinely has 0 citations, while a
 * 2015 paper whose cohort the corpus never crawled has no field-normalized impact at
 * all. Rendering the second as "0x the field median" would state something false, so
 * anything reading these must branch on null rather than defaulting it. */
export interface PaperMetrics {
  paper_id: string;
  title: string;
  citation_count: number | null;
  influential_citation_count: number | null;
  influential_ratio: number | null;
  /** Citations relative to the median paper of the same field and year. */
  cnci: number | null;
  cnci_percentile: number | null;
  pagerank: number | null;
  pagerank_percentile: number | null;
  /** How many papers inside this corpus cite it. */
  in_degree: number | null;
  venue: string | null;
  cohort_year: number | null;
  primary_category: string | null;
  /** One human-readable sentence assembled server-side from whichever fields exist. */
  summary: string;
}

export function getPaperMetrics(paperId: string): Promise<PaperMetrics> {
  return request<PaperMetrics>(`/papers/${encodeURIComponent(paperId)}/metrics`);
}

/** Scorecards for a whole result list in one round trip. Papers the corpus does not
 * hold are omitted from the response, so callers must match on paper_id rather than
 * assuming the array lines up with what they asked for. */
export function getMetricsBatch(paperIds: string[]): Promise<PaperMetrics[]> {
  return postJSON<PaperMetrics[]>("/papers/metrics", { paper_ids: paperIds });
}
