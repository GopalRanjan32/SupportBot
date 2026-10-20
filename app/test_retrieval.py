import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types


CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "notenest_docs"


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
        model="gemini-embedding-2",
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=768
        ),
    )

    return response.embeddings[0].values


def get_collection():
    """
    Open the existing local ChromaDB collection.
    """
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

    return collection


def retrieve(question, top_k=3):
    """
    Retrieve the top-k most relevant chunks
    for a user question.
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

    return results


def print_results(question, results):
    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):
        print(f"\nRESULT #{rank}")
        print(f"Source: {metadata['source']}")
        print(f"Chunk: {metadata['chunk_number']}")
        print(f"Distance: {distance:.4f}")
        print("Text:")
        print(document)
        print("-" * 70)


def main():
    test_questions = [
        "What happens if my Pro payment fails?",
        "How can I export all of my notes?",
        "What happens if I forget my password?",
    ]

    for question in test_questions:
        results = retrieve(
            question=question,
            top_k=3,
        )

        print_results(
            question=question,
            results=results,
        )


if __name__ == "__main__":
    main()