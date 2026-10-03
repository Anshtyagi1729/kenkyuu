from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.routes.schemas import Paper, paper_from_row
from app.storage import db

router = APIRouter()


class Folder(BaseModel):
    folder_id: int
    name: str
    created_at: str
    paper_count: int


class CreateFolderRequest(BaseModel):
    name: str


class SaveRequest(BaseModel):
    folder_id: int | None = None  # None = the default "Want to read" folder


@router.get("/folders", response_model=list[Folder])
def list_folders() -> list[Folder]:
    return [Folder(**row) for row in db.list_folders()]


@router.post("/folders", response_model=Folder)
def create_folder(request: CreateFolderRequest) -> Folder:
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="folder name can't be empty")
    db.create_folder(name)
    folder = next((f for f in db.list_folders() if f["name"] == name), None)
    return Folder(**folder)  # type: ignore[arg-type]


@router.delete("/folders/{folder_id}")
def delete_folder(folder_id: int) -> dict:
    db.delete_folder(folder_id)
    return {"status": "ok"}


@router.get("/folders/{folder_id}/papers", response_model=list[Paper])
def get_folder_papers(folder_id: int) -> list[Paper]:
    return [paper_from_row(row) for row in db.list_folder_papers(folder_id)]


@router.post("/papers/{paper_id}/save")
def save_paper(paper_id: str, request: SaveRequest) -> dict:
    if db.get_paper(paper_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown paper_id: {paper_id}")
    folder_id = request.folder_id or db.get_or_create_default_folder()
    db.save_paper_to_folder(paper_id, folder_id)
    return {"status": "ok", "folder_id": folder_id}


@router.delete("/papers/{paper_id}/save")
def unsave_paper(paper_id: str, folder_id: int) -> dict:
    db.unsave_paper(paper_id, folder_id)
    return {"status": "ok"}
