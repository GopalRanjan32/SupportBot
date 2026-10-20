from app.rag import (
    REFUSAL_MESSAGE,
    is_refusal_answer,
    should_refuse_by_distance,
)


def test_refuses_when_distance_is_too_high():
    chunks = [
        {
            "distance": 0.90,
            "text": "irrelevant text",
            "source": "example.md",
            "chunk_number": 1,
        }
    ]

    assert should_refuse_by_distance(chunks) is True


def test_accepts_when_distance_is_good():
    chunks = [
        {
            "distance": 0.30,
            "text": "relevant text",
            "source": "billing.md",
            "chunk_number": 1,
        }
    ]

    assert should_refuse_by_distance(chunks) is False


def test_refuses_when_no_chunks_exist():
    assert should_refuse_by_distance([]) is True


def test_detects_exact_refusal_message():
    assert is_refusal_answer(
        REFUSAL_MESSAGE
    ) is True


def test_normal_answer_is_not_refusal():
    answer = (
        "The Pro Plan costs $8 per month."
    )

    assert is_refusal_answer(answer) is False