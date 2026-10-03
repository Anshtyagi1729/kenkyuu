from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.llm.agent import answer_query_stream
from app.llm.router import NoProviderAvailable
from app.routes.schemas import source_from_dict
from app.routes.sse import sse_event
from app.storage import db

router = APIRouter()


class Message(BaseModel):
    role: str
    content: str


class SearchRequest(BaseModel):
    history: list[Message]


@router.post("/search")
def search(request: SearchRequest) -> StreamingResponse:
    if not request.history or request.history[-1].role != "user":
        raise HTTPException(status_code=400, detail="history must end with a user message")

    db.log_search(request.history[-1].content)
    history = [m.model_dump() for m in request.history]

    def event_stream():
        try:
            for kind, payload in answer_query_stream(history):
                if kind == "activity":
                    # What the agent is doing right now, in plain language. A real
                    # question takes tens of seconds of tool calls before the first
                    # answer token exists, and this is the only thing the user can be
                    # shown during it that is true.
                    yield sse_event("activity", payload)
                elif kind == "token":
                    yield sse_event("token", payload)
                else:
                    sources = [source_from_dict(s).model_dump() for s in payload]
                    yield sse_event("done", sources)
        except NoProviderAvailable as exc:
            # Headers (200 OK) are already sent by the time a generator can raise -
            # an SSE error event is the only way left to tell the client this failed.
            yield sse_event("error", str(exc))

    return StreamingResponse(event_stream(), media_type="text/event-stream")
