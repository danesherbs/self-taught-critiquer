from datasets import *


def test_next_rationale_step_is_correct_for_zeroed_inputs():
    # Given
    rs = RationaleStep(x=0, y=0, acc="", carry=1, step=0, n_steps=2)
    expected = RationaleStep(x=0, y=0, acc="1", carry=0, step=1, n_steps=2)

    # When
    actual = next_rational_step(rs)

    # Then
    assert actual == expected


def test_next_rationale_step_is_correct_penultimate_step():
    # Given
    rs = RationaleStep(x=0, y=0, acc="86", carry=0, step=2, n_steps=4)
    expected = RationaleStep(x=0, y=0, acc="086", carry=0, step=3, n_steps=4)

    # When
    actual = next_rational_step(rs)

    # Then
    assert actual == expected


def test_rationale_to_str_is_correct_for_zeroed_inputs():
    # Given
    r = [
        RationaleStep(x=0, y=0, acc="86", carry=0, step=2, n_steps=4),
        RationaleStep(x=0, y=0, acc="086", carry=0, step=3, n_steps=4),
    ]
    expected = ", 8 6 C: 0\n0 8 6"

    # When
    actual = rationale_to_str(r)

    # Then
    assert actual == expected


def test_corrected_answer_gives_correct_answer_for_many_inputs():
    for n_digits in range(1, 30):
        for _ in range(10):
            # Given
            ce = make_corrupted_example(default_n_digits=n_digits)
            expected = ce.question.x + ce.question.y

            # When
            actual = ce.correction_answer

            # Then
            assert actual == expected, f"Failed for example\n{corrupted_example_to_str(ce)}"

def test_rationale_step_to_str_on_penultimate_step_with_zero_carry():
    # Given
    rs = RationaleStep(x=0, y=0, acc="29", carry=0, step=2, n_steps=4)
    expected = ", 2 9 C: 0"

    # When
    actual = rationale_step_to_str(rs)

    # Then
    assert actual == expected


def test_rationale_step_to_str_on_penultimate_step_with_one_carry():
    # Given
    rs = RationaleStep(x=0, y=0, acc="29", carry=1, step=2, n_steps=4)
    expected = ", 2 9 C: 1"

    # When
    actual = rationale_step_to_str(rs)

    # Then
    assert actual == expected


def test_rationale_step_to_str_on_final_step():
    # Given
    rs = RationaleStep(x=0, y=0, acc="086", carry=0, step=4, n_steps=5)
    expected = "0 8 6"

    # When
    actual = rationale_step_to_str(rs)

    # Then
    assert actual == expected


def test_rationale_is_corrupt_when_is_corrupt_is_set_to_true():
    for _ in range(100):
        # Given
        q = make_question(n_digits=1)
        r = make_rationale(q, is_corrupted=True)

        # When
        c = correct_rationale(q, r)

        # Then
        assert c is not None, f"Question {q} had a generated rationale {r}"


def test_few_shot_dataset_is_deterministic():
    # Given

    # When
    first_examples = generate_few_shot_examples(n_examples=10, min_n_digits=1, max_n_digits=10)
    second_examples = generate_few_shot_examples(n_examples=10, min_n_digits=1, max_n_digits=10)

    # Then
    assert first_examples == second_examples


def test_arithmetic_dataset_is_deterministic():
    # Given

    # When
    first_examples = list(ArithmeticDataset(n_examples=1_000, min_n_digits=1, max_n_digits=10))
    second_examples = list(ArithmeticDataset(n_examples=1_000, min_n_digits=1, max_n_digits=10))

    # Then
    assert first_examples == second_examples


def test_labelled_arithmetic_dataset_can_produce_corrupted_correction():
    # Given
    rs = RationaleStep(x=0, y=4, acc='', carry=0, step=0, n_steps=3)

    # When
    r = complete_rationale(rs, is_corrupted=True)

    # Then
    assert int(r[-1].acc) != 4
