import os
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types


CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "notenest_docs"

EMBEDDING_MODEL = "gemini-embedding-2"
GENERATION_MODEL = "gemini-3.5-flash-lite"

MAX_RETRIES = 3
REFUSAL_DISTANCE_THRESHOLD = 0.65
REFUSAL_MESSAGE = "I couldn't find this in the documentation."


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Please check your .env file."
    )


gemini_client = genai.Client(api_key=api_key)


def create_embedding(text):
    """
    Convert text into a 768-dimensional Gemini embedding.
    """
    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=768
        ),
    )

    return response.embeddings[0].values


def get_collection():
    """
    Open the existing NoteNest ChromaDB collection.
    """
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    return chroma_client.get_collection(
        name=COLLECTION_NAME
    )


def retrieve(question, top_k=3):
    """
    Retrieve top-k relevant chunks from ChromaDB.
    """
    collection = get_collection()

    question_embedding = create_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    retrieved_chunks = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        retrieved_chunks.append(
            {
                "text": document,
                "source": metadata["source"],
                "chunk_number": metadata["chunk_number"],
                "distance": distance,
            }
        )

    return retrieved_chunks


def build_context(chunks):
    """
    Convert retrieved chunks into a context string for Gemini.
    """
    context_parts = []

    for chunk in chunks:
        context_part = (
            f"[Source: {chunk['source']} | "
            f"Chunk: {chunk['chunk_number']}]\n"
            f"{chunk['text']}"
        )

        context_parts.append(context_part)

    return "\n\n---\n\n".join(context_parts)


def is_temporary_server_error(error):
    """
    Check whether an API error looks temporary.
    """
    error_text = str(error).lower()

    temporary_signals = [
        "503",
        "high demand",
        "service_unavailable",
        "service unavailable",
        "temporarily unavailable",
    ]

    return any(
        signal in error_text
        for signal in temporary_signals
    )


def should_refuse_by_distance(chunks):
    """
    Refuse when retrieval confidence is too weak.

    Lower Chroma distance means better semantic similarity.
    """
    if not chunks:
        return True

    best_distance = chunks[0]["distance"]

    return best_distance > REFUSAL_DISTANCE_THRESHOLD


def generate_answer(question, chunks):
    """
    Generate an answer from retrieved NoteNest context.
    """
    context = build_context(chunks)

    prompt = f"""
You are SupportBot, the customer support assistant for NoteNest.

Answer the user's question using ONLY the NoteNest documentation
provided in the context below.

Rules:
1. Do not use outside knowledge.
2. Do not invent NoteNest features, policies, prices, or procedures.
3. Give a concise and helpful answer.
4. Do not add citations inside the answer text.
5. Citations will be displayed separately by the application.
6. If the documentation does not clearly answer the question,
   say exactly:
   "{REFUSAL_MESSAGE}"

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
""".strip()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            interaction = gemini_client.interactions.create(
                model=GENERATION_MODEL,
                input=prompt,
            )

            answer = interaction.output_text

            if not answer:
                raise ValueError(
                    "Gemini returned an empty response."
                )

            return answer.strip()

        except Exception as error:
            if not is_temporary_server_error(error):
                raise

            print(
                f"Gemini temporary server error on attempt "
                f"{attempt}/{MAX_RETRIES}: {error}"
            )

            if attempt == MAX_RETRIES:
                raise

            wait_seconds = 2 ** attempt

            print(
                f"Retrying after {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)


def is_refusal_answer(answer):
    """
    Check whether Gemini itself decided that
    the documentation does not contain the answer.
    """
    normalized_answer = answer.strip().lower()
    normalized_refusal = REFUSAL_MESSAGE.lower()

    return normalized_answer == normalized_refusal


def answer_question(question):
    """
    Complete RAG pipeline with two-layer refusal logic:
    1. Retrieval distance threshold
    2. Gemini context-grounded refusal
    """
    chunks = retrieve(
        question=question,
        top_k=3,
    )

    best_distance = (
        chunks[0]["distance"]
        if chunks
        else None
    )

    if should_refuse_by_distance(chunks):
        return {
            "question": question,
            "answer": REFUSAL_MESSAGE,
            "citations": [],
            "refused": True,
            "refusal_reason": "distance_threshold",
            "best_distance": best_distance,
        }

    answer = generate_answer(
        question=question,
        chunks=chunks,
    )

    if is_refusal_answer(answer):
        return {
            "question": question,
            "answer": REFUSAL_MESSAGE,
            "citations": [],
            "refused": True,
            "refusal_reason": "llm_context_check",
            "best_distance": best_distance,
        }

    citations = []

    for chunk in chunks:
        citations.append(
            {
                "source": chunk["source"],
                "chunk_number": chunk["chunk_number"],
                "distance": chunk["distance"],
            }
        )

    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "refused": False,
        "refusal_reason": None,
        "best_distance": best_distance,
    }


def main():
    question = input("Ask NoteNest: ").strip()

    if not question:
        print("Please enter a question.")
        return

    result = answer_question(question)

    print("\nANSWER")
    print("-" * 70)
    print(result["answer"])

    print("\nDEBUG")
    print("-" * 70)

    if result["best_distance"] is not None:
        print(
            f"Best retrieval distance: "
            f"{result['best_distance']:.4f}"
        )

    print(
        f"Refusal threshold: "
        f"{REFUSAL_DISTANCE_THRESHOLD:.4f}"
    )

    print(
        f"Refused: {result['refused']}"
    )

    print(
        f"Refusal reason: {result['refusal_reason']}"
    )

    if result["citations"]:
        print("\nSOURCES")
        print("-" * 70)

        for citation in result["citations"]:
            print(
                f"- {citation['source']} "
                f"(chunk {citation['chunk_number']}, "
                f"distance {citation['distance']:.4f})"
            )


if __name__ == "__main__":
    main()