"use client";

import { useEffect, useState } from "react";
import PaperCard from "@/components/PaperCard";
import { createFolder, deleteFolder, getFolderPapers, listFolders, type Folder, type Paper } from "@/lib/api";
import { TrashIcon } from "@/components/icons";

export default function LibraryPage() {
  const [folders, setFolders] = useState<Folder[] | null>(null);
  const [selectedFolderId, setSelectedFolderId] = useState<number | null>(null);
  const [loadedPapers, setLoadedPapers] = useState<{ folderId: number; papers: Paper[] } | null>(null);
  const [newFolderName, setNewFolderName] = useState("");

  useEffect(() => {
    listFolders().then((loaded) => {
      setFolders(loaded);
      setSelectedFolderId((current) => current ?? loaded[0]?.folder_id ?? null);
    });
  }, []);

  // Same id-pairing as the paper pages: no synchronous clear inside the effect, and
  // switching folders shows the previous folder's papers only until the new ones
  // arrive rather than blanking the list on every click.
  useEffect(() => {
    if (selectedFolderId === null) return;
    let cancelled = false;
    getFolderPapers(selectedFolderId).then(
      (list) => !cancelled && setLoadedPapers({ folderId: selectedFolderId, papers: list }),
    );
    return () => {
      cancelled = true;
    };
  }, [selectedFolderId]);

  const papers = loadedPapers?.folderId === selectedFolderId ? loadedPapers.papers : null;

  async function handleCreateFolder(e: React.FormEvent) {
    e.preventDefault();
    const name = newFolderName.trim();
    if (!name) return;
    const folder = await createFolder(name);
    setFolders((prev) => (prev ? [...prev, folder] : [folder]));
    setNewFolderName("");
    setSelectedFolderId(folder.folder_id);
  }

  async function handleDeleteFolder(folderId: number) {
    if (!window.confirm("Delete this folder? Saved papers stay in your library, just removed from this folder.")) {
      return;
    }
    await deleteFolder(folderId);
    setFolders((prev) => {
      const next = prev?.filter((f) => f.folder_id !== folderId) ?? null;
      if (selectedFolderId === folderId) setSelectedFolderId(next?.[0]?.folder_id ?? null);
      return next;
    });
  }

  return (
    <main className="mx-auto flex w-full min-h-0 max-w-4xl flex-1 gap-8 px-6 py-8">
      <aside className="scroll-area flex w-56 shrink-0 flex-col gap-1 overflow-y-auto">
        <h1 className="mb-3 font-serif text-2xl font-semibold text-foreground">Library</h1>

        {folders?.length === 0 && (
          <p className="mb-2 text-sm text-foreground-muted">Create a folder to start saving papers.</p>
        )}

        {folders?.map((folder) => (
          <div key={folder.folder_id} className="group flex items-center gap-0.5">
            <button
              onClick={() => setSelectedFolderId(folder.folder_id)}
              className={`min-w-0 flex-1 truncate rounded-lg px-3 py-2 text-left text-sm transition-colors ${
                selectedFolderId === folder.folder_id
                  ? "bg-accent-soft font-medium text-accent-hover"
                  : "text-foreground-muted hover:bg-surface-muted"
              }`}
            >
              {folder.name} <span className="text-xs opacity-60">({folder.paper_count})</span>
            </button>
            <button
              onClick={() => handleDeleteFolder(folder.folder_id)}
              aria-label="Delete folder"
              className="shrink-0 rounded-full p-1.5 text-foreground-muted opacity-0 transition-all hover:text-red-600 group-hover:opacity-100"
            >
              <TrashIcon />
            </button>
          </div>
        ))}

        <form onSubmit={handleCreateFolder} className="mt-2 flex gap-1">
          <input
            value={newFolderName}
            onChange={(e) => setNewFolderName(e.target.value)}
            placeholder="New folder"
            className="min-w-0 flex-1 rounded-md border border-border bg-surface px-2 py-1.5 text-xs outline-none transition-shadow focus:border-accent focus:ring-4 focus:ring-accent/10"
          />
          <button
            type="submit"
            disabled={!newFolderName.trim()}
            className="shrink-0 rounded-md border border-border px-2 py-1.5 text-xs font-medium text-foreground-muted transition-colors hover:border-accent hover:text-accent disabled:opacity-40"
          >
            Add
          </button>
        </form>
      </aside>

      <div className="scroll-area flex min-h-0 min-w-0 flex-1 flex-col gap-2 overflow-y-auto pr-2">
        {selectedFolderId === null ? (
          <p className="text-sm text-foreground-muted">No folder selected.</p>
        ) : papers === null ? null : papers.length === 0 ? (
          <p className="text-sm text-foreground-muted">No papers saved here yet.</p>
        ) : (
          // Rendered through PaperCard rather than bespoke markup, so a saved paper
          // carries the same impact badges it had in the results that led here.
          // Reviewing a reading list is exactly when "how influential is this" is the
          // question, and the library was the one place that dropped the answer.
          papers.map((paper) => <PaperCard key={paper.paper_id} paperId={paper.paper_id} />)
        )}
      </div>
    </main>
  );
}
