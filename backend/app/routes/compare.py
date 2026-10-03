from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.llm.compare import compare_papers
from app.llm.router import NoProviderAvailable
from app.routes.schemas import Source, source_from_dict

router = APIRouter()


class CompareRequest(BaseModel):
    paper_ids: list[str]
    focus: str | None = None


class CompareResponse(BaseModel):
    comparison: str
    sources: list[Source]


@router.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest) -> CompareResponse:
    try:
        result = compare_papers(request.paper_ids, focus=request.focus)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NoProviderAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return CompareResponse(comparison=result.text, sources=[source_from_dict(s) for s in result.sources])
