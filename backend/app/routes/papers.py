from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.ingestion import citation_graph
from app.llm.paper_chat import chat_about_paper_stream
from app.llm.router import NoProviderAvailable
from app.routes.schemas import Paper, PaperMetrics, paper_from_row, source_from_dict
from app.routes.sse import sse_event
from app.scientometrics import scorecard
from app.storage import db

router = APIRouter()


class Message(BaseModel):
    role: str
    content: str


class PaperChatRequest(BaseModel):
    history: list[Message]


class GraphNode(BaseModel):
    paper_id: str
    title: str
    year: int | None = None


class GraphEdge(BaseModel):
    source: str
    target: str


class CitationGraph(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


@router.get("/papers/{paper_id}", response_model=Paper)
def get_paper(paper_id: str) -> Paper:
    paper = db.get_paper(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"unknown paper_id: {paper_id}")
    return paper_from_row(paper)


@router.get("/papers/{paper_id}/citation-graph", response_model=CitationGraph)
def get_citation_graph(paper_id: str) -> CitationGraph:
    try:
        graph = citation_graph.get_graph(paper_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return CitationGraph(**graph)


class MetricsRequest(BaseModel):
    paper_ids: list[str]


@router.post("/papers/metrics", response_model=list[PaperMetrics])
def get_metrics_batch(request: MetricsRequest) -> list[PaperMetrics]:
    """Scorecards for a whole result list in one call.

    POST rather than GET with a query string: a search returns up to ten arXiv ids and
    putting them in the URL runs into length limits and encoding noise for no benefit.
    Registered before /papers/{paper_id} matters too - FastAPI matches in declaration
    order, and the parameterized route would otherwise swallow the literal "metrics".

    Papers absent from the corpus are simply omitted rather than erroring: a result
    list can legitimately reference a paper discovered live from arXiv that was never
    ingested, and one such entry should not fail the badges for the other nine.
    """
    cards = scorecard.get_many(request.paper_ids)
    return [PaperMetrics(**cards[pid].to_dict()) for pid in request.paper_ids if pid in cards]


@router.get("/papers/{paper_id}/metrics", response_model=PaperMetrics)
def get_paper_metrics(paper_id: str) -> PaperMetrics:
    card = scorecard.get(paper_id)
    if card is None:
        raise HTTPException(status_code=404, detail=f"unknown paper_id: {paper_id}")
    return PaperMetrics(**card.to_dict())


@router.get("/papers/{paper_id}/folders", response_model=list[int])
def get_paper_folders(paper_id: str) -> list[int]:
    return db.list_paper_folder_ids(paper_id)


@router.post("/papers/{paper_id}/chat")
def paper_chat(paper_id: str, request: PaperChatRequest) -> StreamingResponse:
    if not request.history or request.history[-1].role != "user":
        raise HTTPException(status_code=400, detail="history must end with a user message")
    # Checked up front rather than inside the generator: once streaming starts the
    # response is already committed to 200 OK, so a 404 has to happen before that.
    if db.get_paper(paper_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown paper_id: {paper_id}")

    history = [m.model_dump() for m in request.history]

    def event_stream():
        try:
            for kind, payload in chat_about_paper_stream(paper_id, history):
                if kind == "token":
                    yield sse_event("token", payload)
                else:
                    sources = [source_from_dict(s).model_dump() for s in payload]
                    yield sse_event("done", sources)
        except NoProviderAvailable as exc:
            yield sse_event("error", str(exc))

    return StreamingResponse(event_stream(), media_type="text/event-stream")
