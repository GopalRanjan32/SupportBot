# SupportBot

SupportBot is a RAG-based customer support assistant built for the fictional note-taking product **NoteNest**.

It retrieves relevant information from product documentation, generates grounded answers using Gemini, shows source citations, and refuses unsupported questions instead of hallucinating.

## Features

- Retrieval-Augmented Generation (RAG)
- Gemini embeddings
- ChromaDB vector search
- Grounded answers from product documentation
- Source file and chunk citations
- Two-layer refusal logic
- FastAPI backend
- HTML, CSS, and vanilla JavaScript frontend
- Evaluation pipeline
- Local persistent vector database

## Tech Stack

- Python
- FastAPI
- Uvicorn
- Google Gemini API
- `google-genai`
- ChromaDB
- HTML
- CSS
- JavaScript

## How It Works

```text
User Question
      |
      v
Gemini Embedding
      |
      v
ChromaDB Retrieval
      |
      v
Relevant Documentation Chunks
      |
      v
Refusal Check
      |
      v
Gemini Grounded Generation
      |
      v
Answer + Source Citations

Hallucination Control
SupportBot uses two layers of refusal logic.
1. Retrieval Distance Threshold
If the retrieved documentation is too dissimilar to the user's question, the system refuses to answer.
2. LLM Context Check
Even when a retrieved chunk is semantically similar, Gemini is instructed to answer only when the supplied documentation clearly supports the answer.

If the documentation does not contain the answer, SupportBot responds:
I couldn't find this in the documentation.

No citations are displayed for refused questions.
Evaluation
The system was evaluated using a custom dataset of 25 questions:
- 20 answerable NoteNest support questions
- 5 out-of-scope questions
Current internal evaluation results:
Metric	Result
Answer accuracy	25/25 (100%)
Citation accuracy	25/25 (100%)
Refusal accuracy	25/25 (100%)


These results are based on a small controlled evaluation dataset and should not be interpreted as general real-world accuracy.
Example
Question:
What happens if my Pro payment fails?

Answer:
If a Pro subscription payment fails, NoteNest retries the payment for 7 days.
During this period, the user can continue using Pro features.
If payment is still unsuccessful after 7 days, the account is moved to the Free Plan.

Source:
billing.md - chunk 4

Out-of-scope question:
Does NoteNest support cryptocurrency trading?

Response:
I couldn't find this in the documentation.

Project Status
Current completed components:
- Product documentation
- Document loading and chunking
- Gemini embeddings
- ChromaDB vector store
- Semantic retrieval
- RAG generation
- Citation handling
- Hallucination/refusal logic
- FastAPI API
- Web chat interface
- Evaluation dataset
- Evaluation script
Upcoming:
- Automated tests
- Docker
- GitHub Actions CI
- Render deployment
