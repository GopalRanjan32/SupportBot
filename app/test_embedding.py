import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Please check your .env file."
    )


client = genai.Client(api_key=api_key)


def create_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=768
        ),
    )

    return response.embeddings[0].values


def main():
    sample_text = "How do I cancel my NoteNest subscription?"

    embedding = create_embedding(sample_text)

    print("Text:")
    print(sample_text)

    print("\nEmbedding created successfully.")
    print(f"Embedding dimensions: {len(embedding)}")

    print("\nFirst 10 values:")
    print(embedding[:10])


if __name__ == "__main__":
    main()