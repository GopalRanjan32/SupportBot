from rag import answer_question


def print_result(result):
    print("\n" + "=" * 80)
    print(f"QUESTION: {result['question']}")
    print("=" * 80)

    print("\nANSWER:")
    print(result["answer"])

    print("\nSTATUS:")
    print(f"Refused: {result['refused']}")
    print(f"Refusal reason: {result['refusal_reason']}")

    if result["best_distance"] is not None:
        print(
            f"Best distance: "
            f"{result['best_distance']:.4f}"
        )

    if result["citations"]:
        print("\nSOURCES:")

        for citation in result["citations"]:
            print(
                f"- {citation['source']} "
                f"(chunk {citation['chunk_number']}, "
                f"distance {citation['distance']:.4f})"
            )

    print()


def main():
    questions = [
        "What happens if my Pro payment fails?",
        "How can I export all my notes?",
        "How do I reset my password?",
        "What is the weather in Delhi today?",
        "Does NoteNest offer cryptocurrency trading?",
    ]

    for question in questions:
        result = answer_question(question)
        print_result(result)


if __name__ == "__main__":
    main()