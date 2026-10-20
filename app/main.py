from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.rag import answer_question


STATIC_DIR = Path("static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Build the ChromaDB index from NoteNest docs
    whenever the application starts.
    """
    print("Building NoteNest vector index...")

    from app.ingest import (
        load_markdown_files,
        create_chunks,
        build_vector_store,
    )

    documents = load_markdown_files()
    chunks = create_chunks(documents)

    collection = build_vector_store(chunks)

    print(
        f"Vector index ready. "
        f"Stored chunks: {collection.count()}"
    )

    yield

    print("SupportBot shutting down.")


app = FastAPI(
    title="SupportBot API",
    description=(
        "RAG-based customer support assistant "
        "for the fictional NoteNest product."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=500,
        description="Question about NoteNest",
    )


@app.get("/")
def home():
    """
    Serve the SupportBot web interface.
    """
    return FileResponse(
        STATIC_DIR / "index.html"
    )


@app.get("/health")
def health():
    """
    Simple health check endpoint.
    """
    return {
        "status": "ok",
    }


@app.post("/ask")
def ask(request: AskRequest):
    """
    Run the complete RAG pipeline for a user question.
    """
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        result = answer_question(
            question
        )

        return {
            "question":
                result["question"],

            "answer":
                result["answer"],

            "refused":
                result["refused"],

            "refusal_reason":
                result["refusal_reason"],

            "citations":
                result["citations"],
        }

    except Exception as error:
        print(
            "Error while processing "
            f"question: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to process "
                "the question right now."
            ),
        )