import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from app.rag import answer_question


DATASET_PATH = (
    PROJECT_ROOT
    / "eval"
    / "eval_dataset.json"
)


def load_dataset():
    """
    Load the evaluation dataset from JSON.
    """
    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def citation_is_correct(
    result,
    expected_source,
):
    """
    Check whether the expected source appears
    in the returned citations.
    """
    if expected_source is None:
        return True

    returned_sources = [
        citation["source"]
        for citation in result["citations"]
    ]

    return expected_source in returned_sources


def evaluate_case(case):
    """
    Run one evaluation case and compare
    actual behavior with expected behavior.
    """
    result = answer_question(
        case["question"]
    )

    expected_refusal = (
        case["expected_refusal"]
    )

    actual_refusal = result["refused"]

    refusal_correct = (
        actual_refusal
        == expected_refusal
    )

    if expected_refusal:
        answer_correct = actual_refusal
    else:
        answer_correct = (
            not actual_refusal
            and bool(
                result["answer"].strip()
            )
        )

    if expected_refusal:
        citation_correct = (
            len(result["citations"]) == 0
        )
    else:
        citation_correct = (
            citation_is_correct(
                result,
                case["expected_source"],
            )
        )

    return {
        "id": case["id"],
        "question": case["question"],
        "expected_refusal":
            expected_refusal,
        "actual_refusal":
            actual_refusal,
        "answer_correct":
            answer_correct,
        "citation_correct":
            citation_correct,
        "refusal_correct":
            refusal_correct,
        "answer":
            result["answer"],
        "citations":
            result["citations"],
    }


def percentage(
    correct,
    total,
):
    """
    Convert a count into a percentage.
    """
    if total == 0:
        return 0.0

    return (
        correct / total
    ) * 100


def main():
    dataset = load_dataset()

    results = []

    print(
        f"Running evaluation on "
        f"{len(dataset)} questions...\n"
    )

    for index, case in enumerate(
        dataset,
        start=1,
    ):
        print(
            f"[{index}/{len(dataset)}] "
            f"{case['question']}"
        )

        result = evaluate_case(case)

        results.append(result)

        print(
            f"  Refused: "
            f"{result['actual_refusal']}"
        )

        print(
            f"  Answer correct: "
            f"{result['answer_correct']}"
        )

        print(
            f"  Citation correct: "
            f"{result['citation_correct']}"
        )

        print(
            f"  Refusal correct: "
            f"{result['refusal_correct']}"
        )

        print()


    total = len(results)

    answer_correct_count = sum(
        result["answer_correct"]
        for result in results
    )

    citation_correct_count = sum(
        result["citation_correct"]
        for result in results
    )

    refusal_correct_count = sum(
        result["refusal_correct"]
        for result in results
    )


    answer_accuracy = percentage(
        answer_correct_count,
        total,
    )

    citation_accuracy = percentage(
        citation_correct_count,
        total,
    )

    refusal_accuracy = percentage(
        refusal_correct_count,
        total,
    )


    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Total questions: {total}"
    )

    print(
        f"Answer accuracy: "
        f"{answer_correct_count}/{total} "
        f"({answer_accuracy:.2f}%)"
    )

    print(
        f"Citation accuracy: "
        f"{citation_correct_count}/{total} "
        f"({citation_accuracy:.2f}%)"
    )

    print(
        f"Refusal accuracy: "
        f"{refusal_correct_count}/{total} "
        f"({refusal_accuracy:.2f}%)"
    )


    output_path = (
        PROJECT_ROOT
        / "eval"
        / "eval_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "summary": {
                    "total_questions":
                        total,
                    "answer_accuracy":
                        round(
                            answer_accuracy,
                            2,
                        ),
                    "citation_accuracy":
                        round(
                            citation_accuracy,
                            2,
                        ),
                    "refusal_accuracy":
                        round(
                            refusal_accuracy,
                            2,
                        ),
                },
                "results":
                    results,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )


    print(
        "\nDetailed results saved to:"
    )

    print(output_path)


if __name__ == "__main__":
    main()