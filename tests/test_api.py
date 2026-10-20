from fastapi.testclient import TestClient

import app.main as main_module


client = TestClient(
    main_module.app
)


def test_health_endpoint():
    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok"
    }


def test_ask_endpoint_with_mocked_rag(
    monkeypatch,
):
    def fake_answer_question(question):
        return {
            "question": question,
            "answer": (
                "The Pro Plan costs "
                "$8 per month."
            ),
            "refused": False,
            "refusal_reason": None,
            "citations": [
                {
                    "source": "billing.md",
                    "chunk_number": 2,
                    "distance": 0.25,
                }
            ],
        }

    monkeypatch.setattr(
        main_module,
        "answer_question",
        fake_answer_question,
    )

    response = client.post(
        "/ask",
        json={
            "question":
                "How much is the Pro Plan?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["refused"] is False

    assert data["answer"] == (
        "The Pro Plan costs $8 per month."
    )

    assert (
        data["citations"][0]["source"]
        == "billing.md"
    )


def test_ask_endpoint_rejects_empty_question():
    response = client.post(
        "/ask",
        json={
            "question": ""
        },
    )

    assert response.status_code == 422