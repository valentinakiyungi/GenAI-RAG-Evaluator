"""REST API for the GenAI RAG Evaluator.

    uvicorn api:app --reload      # docs at http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.config import SEARCH_TYPES, ConfigError, get_settings, has_api_key
from src.evaluate import run_ragas_evaluation
from src.history import log_run
from src.ingest import ingest_paths, save_upload
from src.pipeline import answer_question, evaluate_result
from src.retriever import delete_source, list_sources

app = FastAPI(
    title="GenAI RAG Evaluator",
    description="Document Q&A with automated RAGAS evaluation.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    search_type: str | None = None
    ground_truth: str = ""
    evaluate: bool = False


class EvaluateRequest(BaseModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    contexts: list[str] = Field(min_length=1)
    ground_truth: str = ""


def _guard(action):
    """Run `action`, translating expected failures into HTTP errors."""
    try:
        return action()
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "openai_api_key_configured": has_api_key()}


@app.get("/documents")
def documents() -> dict:
    sources = _guard(list_sources)
    return {"documents": [{"source": s, "chunks": n} for s, n in sources.items()]}


@app.post("/documents")
async def upload_documents(files: list[UploadFile] = File(...)) -> dict:
    settings = get_settings()
    payloads = [(file.filename or "", await file.read()) for file in files]

    def action():
        paths = [save_upload(name, content, settings) for name, content in payloads]
        return ingest_paths(paths, settings)

    return {"results": [report.__dict__ for report in _guard(action)]}


@app.delete("/documents/{source}")
def remove_document(source: str) -> dict:
    removed = _guard(lambda: delete_source(source))
    if not removed:
        raise HTTPException(status_code=404, detail=f"No indexed document named {source!r}")
    return {"source": source, "chunks_removed": removed}


@app.post("/query")
def query(request: QueryRequest) -> dict:
    if request.search_type and request.search_type not in SEARCH_TYPES:
        raise HTTPException(status_code=400, detail=f"search_type must be one of {SEARCH_TYPES}")
    settings = get_settings()

    def action():
        result = answer_question(
            request.question, request.top_k, request.search_type, request.ground_truth, settings
        )
        if request.evaluate and result.contexts:
            evaluate_result(result, settings)
        log_run(settings.history_file, result.to_dict())
        return result.to_dict()

    return _guard(action)


@app.post("/evaluate")
def evaluate(request: EvaluateRequest) -> dict:
    scores = _guard(
        lambda: run_ragas_evaluation(
            request.question, request.answer, request.contexts, request.ground_truth
        )
    )
    return {"scores": scores}
