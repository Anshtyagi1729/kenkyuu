"use client";

import { useEffect, useRef, useState } from "react";
import {
  createFolder,
  getPaperFolders,
  listFolders,
  savePaperToFolder,
  unsavePaperFromFolder,
  type Folder,
} from "@/lib/api";
import { BookmarkIcon, CheckIcon } from "@/components/icons";

export default function SaveButton({ paperId }: { paperId: string }) {
  const [open, setOpen] = useState(false);
  const [folders, setFolders] = useState<Folder[] | null>(null);
  const [savedIn, setSavedIn] = useState<Set<number>>(new Set());
  const [newFolderName, setNewFolderName] = useState("");
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getPaperFolders(paperId)
      .then((ids) => setSavedIn(new Set(ids)))
      .catch(() => {});
  }, [paperId]);

  useEffect(() => {
    if (!open) return;
    listFolders()
      .then(setFolders)
      .catch(() => {});
  }, [open]);

  useEffect(() => {
    if (!open) return;
    function handlePointerDown(e: PointerEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setOpen(false);
    }
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKey);
    };
  }, [open]);

  async function toggleFolder(folderId: number) {
    if (savedIn.has(folderId)) {
      await unsavePaperFromFolder(paperId, folderId);
      setSavedIn((prev) => {
        const next = new Set(prev);
        next.delete(folderId);
        return next;
      });
    } else {
      await savePaperToFolder(paperId, folderId);
      setSavedIn((prev) => new Set(prev).add(folderId));
    }
  }

  async function handleCreateFolder(e: React.FormEvent) {
    e.preventDefault();
    const name = newFolderName.trim();
    if (!name) return;
    const folder = await createFolder(name);
    setFolders((prev) => (prev ? [...prev, folder] : [folder]));
    await savePaperToFolder(paperId, folder.folder_id);
    setSavedIn((prev) => new Set(prev).add(folder.folder_id));
    setNewFolderName("");
  }

  const isSaved = savedIn.size > 0;

  return (
    <div ref={menuRef} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className={`flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-sm font-medium transition-colors ${
          isSaved
            ? "border-accent bg-accent-soft text-accent-hover"
            : "border-border text-foreground-muted hover:border-accent hover:text-accent"
        }`}
      >
        <BookmarkIcon filled={isSaved} />
        {isSaved ? "Saved" : "Save"}
      </button>

      {open && (
        <div className="absolute top-full right-0 z-10 mt-1 w-56 overflow-hidden rounded-xl border border-border bg-surface py-1 shadow-[0_8px_24px_-8px_rgba(38,37,33,0.18)]">
          {folders === null && <div className="px-3 py-2 text-sm text-foreground-muted">Loading…</div>}
          {folders?.length === 0 && <div className="px-3 py-2 text-sm text-foreground-muted">No folders yet</div>}
          <div className="scroll-area max-h-52 overflow-y-auto">
            {folders?.map((folder) => (
              <button
                key={folder.folder_id}
                onClick={() => toggleFolder(folder.folder_id)}
                className="flex w-full items-center justify-between px-3 py-2 text-left text-sm text-foreground transition-colors hover:bg-surface-muted"
              >
                <span className="truncate">{folder.name}</span>
                {savedIn.has(folder.folder_id) && <span className="shrink-0 text-accent"><CheckIcon /></span>}
              </button>
            ))}
          </div>
          <form onSubmit={handleCreateFolder} className="flex gap-1 border-t border-border p-1.5">
            <input
              value={newFolderName}
              onChange={(e) => setNewFolderName(e.target.value)}
              placeholder="New folder"
              className="min-w-0 flex-1 rounded-md border border-border bg-background px-2 py-1 text-xs outline-none focus:border-accent"
            />
            <button
              type="submit"
              disabled={!newFolderName.trim()}
              className="shrink-0 rounded-md bg-accent px-2 py-1 text-xs font-medium text-white transition-colors hover:bg-accent-hover disabled:opacity-40"
            >
              Add
            </button>
          </form>
        </div>
      )}
    </div>
  );
}
