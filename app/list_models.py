import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Please check your .env file."
    )


client = genai.Client(api_key=api_key)


def main():
    print("Models available to this API key that support generateContent:\n")

    count = 0

    for model in client.models.list():
        supported_actions = model.supported_actions or []

        if "generateContent" in supported_actions:
            print(model.name)
            count += 1

    print(f"\nTotal generation models found: {count}")


if __name__ == "__main__":
    main()