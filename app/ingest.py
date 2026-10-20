import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types


DOCS_DIR = Path("docs")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "notenest_docs"


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Please check your .env file."
    )


gemini_client = genai.Client(api_key=api_key)


def load_markdown_files():
    """
    Load all Markdown files from the docs folder.
    """
    documents = []

    for file_path in sorted(DOCS_DIR.glob("*.md")):
        content = file_path.read_text(encoding="utf-8")

        documents.append(
            {
                "source": file_path.name,
                "content": content,
            }
        )

    return documents


def split_into_sections(text):
    """
    Split Markdown whenever a heading starts.
    """
    sections = []
    current_section = []

    for line in text.splitlines():
        if line.startswith("#") and current_section:
            section_text = "\n".join(current_section).strip()

            if section_text:
                sections.append(section_text)

            current_section = []

        current_section.append(line)

    if current_section:
        section_text = "\n".join(current_section).strip()

        if section_text:
            sections.append(section_text)

    return sections


def create_chunks(documents):
    """
    Convert documents into smaller chunks.
    """
    chunks = []

    for document in documents:
        source = document["source"]
        sections = split_into_sections(document["content"])

        for chunk_number, section in enumerate(sections, start=1):
            chunk_id = f"{source}-{chunk_number}"

            chunks.append(
                {
                    "id": chunk_id,
                    "source": source,
                    "chunk_number": chunk_number,
                    "text": section,
                }
            )

    return chunks


def create_embedding(text):
    """
    Generate a 768-dimensional embedding using Gemini.
    """
    response = gemini_client.models.embed_content(
        model="gemini-embedding-2",
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=768
        ),
    )

    return response.embeddings[0].values


def build_vector_store(chunks):
    """
    Create a persistent ChromaDB collection
    and store chunks with Gemini embeddings.
    """
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    try:
        chroma_client.delete_collection(
            name=COLLECTION_NAME
        )
        print("Old collection deleted.")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME
    )

    for index, chunk in enumerate(chunks, start=1):
        print(
            f"Embedding chunk {index}/{len(chunks)}: "
            f"{chunk['id']}"
        )

        embedding = create_embedding(chunk["text"])

        collection.add(
            ids=[chunk["id"]],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[
                {
                    "source": chunk["source"],
                    "chunk_number": chunk["chunk_number"],
                }
            ],
        )

    return collection


def main():
    documents = load_markdown_files()
    chunks = create_chunks(documents)

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")

    print("\nBuilding ChromaDB vector store...\n")

    collection = build_vector_store(chunks)

    print("\nVector store created successfully.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Stored chunks: {collection.count()}")
    print(f"Database folder: {CHROMA_DIR}")


if __name__ == "__main__":
    main()