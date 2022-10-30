import pytest
import utils
import torch
import transformers


@pytest.fixture
def tokenizer():
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def test_test_collate_fn_returns_correct_prompt_and_expected_critique_for_minimilistic_single_example(tokenizer):
    # Given
    few_shot_examples = "A\n"
    batch = [("B\n", "C")]
    device = "cpu"
    A_tkn, n_tkn, B_tkn, _, C_tkn = tokenizer(few_shot_examples + batch[0][0] + batch[0][1]).input_ids
    prompt_len = 5
    completion_len = 5
    expected_prompts = {
        "input_ids": torch.tensor([tokenizer.pad_token_id, A_tkn, n_tkn, B_tkn, n_tkn], dtype=torch.long),
        "attention_mask": torch.tensor([0, 1, 1, 1, 1], dtype=torch.long),
    }
    expected_expected_critiques = {
        "input_ids": torch.tensor([C_tkn] + [tokenizer.pad_token_id] * (completion_len - 1), dtype=torch.long),
        "attention_mask": torch.tensor([1] + [0] * (completion_len - 1), dtype=torch.long),
    }

    # When
    _, _, _, actual_prompts, actual_expected_critiques = utils.test_collate_fn(
        batch=batch,
        tokenizer=tokenizer,
        prompt_len=prompt_len,
        completion_len=completion_len,
        few_shot_examples=few_shot_examples,
        device=device,
    )

    # Then
    assert torch.all(actual_prompts["input_ids"] == expected_prompts["input_ids"])
    assert torch.all(actual_prompts["attention_mask"] == expected_prompts["attention_mask"])
    assert torch.all(actual_expected_critiques["input_ids"] == expected_expected_critiques["input_ids"])
    assert torch.all(actual_expected_critiques["attention_mask"] == expected_expected_critiques["attention_mask"])


def test_test_collate_fn_returns_correct_input_ids_and_attn_mask_for_minimilistic_single_example(tokenizer):
    # Given
    few_shot_examples = "A\n"
    batch = [("B\n", "C")]
    device = "cpu"
    A_tkn, n_tkn, B_tkn, _, C_tkn = tokenizer(few_shot_examples + batch[0][0] + batch[0][1]).input_ids
    prompt_len = 5
    completion_len = 5
    expected_input_ids = torch.tensor([
    #            x                 A      \n     B     \n      C      x  x  x  x
        [tokenizer.pad_token_id, A_tkn, n_tkn, B_tkn, n_tkn, C_tkn] + [tokenizer.pad_token_id] * (completion_len - 1),
    ], dtype=torch.long)
    expected_attn_mask = torch.tensor([
    #    x  A \n  B \n  C  x  x  x  x
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    ], dtype=torch.long)
    expected_prompt_mask = torch.tensor([
    #    x  A \n  B \n  C  x  x  x  x
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    ], dtype=torch.long)

    # When
    actual_input_ids, actual_attn_mask, actual_prompt_mask, _, _ = utils.test_collate_fn(
        batch=batch,
        tokenizer=tokenizer,
        prompt_len=prompt_len,
        completion_len=completion_len,
        few_shot_examples=few_shot_examples,
        device=device,
    )

    # Then
    assert torch.all(actual_input_ids == expected_input_ids)
    assert torch.all(actual_attn_mask == expected_attn_mask)
    assert torch.all(actual_prompt_mask == expected_prompt_mask)


def test_keep_only_masked_examples_keeps_correct_examples():
    # Given
    prompts = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "attention_mask": torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
    }
    completions = {
        "input_ids": torch.tensor([[3, 2, 1], [6, 5, 4], [9, 8, 7]]),
        "attention_mask": torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    }
    mask = torch.tensor([1, 0, 1], dtype=torch.bool)
    expected_correct_examples = {
        "input_ids": torch.tensor([[1, 2, 3, 3, 2, 1], [7, 8, 9, 9, 8, 7]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 0, 1]]),
        "prompt_mask": torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]]),
    }

    # When
    actual_correct_examples = utils._keep_only_masked_examples(
        prompts=prompts,
        completions=completions,
        mask=mask,
    )
    
    # Then
    assert torch.all(actual_correct_examples["input_ids"] == expected_correct_examples["input_ids"])
    assert torch.all(actual_correct_examples["attention_mask"] == expected_correct_examples["attention_mask"])
    assert torch.all(actual_correct_examples["prompt_mask"] == expected_correct_examples["prompt_mask"])


def test_is_generated_critique_correct_returns_correct_mask(tokenizer):
    # Given
    actual_critique_ids = torch.tensor([
        [1, 2, 3, tokenizer.pad_token_id, tokenizer.pad_token_id, tokenizer.pad_token_id],
        [1, 2, 3, 4, tokenizer.pad_token_id, tokenizer.pad_token_id],
        [1] + [tokenizer.pad_token_id] * 5,
        [9, 9, 9, 9, 9, 9],
    ])
    expected_critique_ids = torch.tensor([
        [1, 2, 3],
        [1, 2, tokenizer.pad_token_id],
        [1, 2, tokenizer.pad_token_id],
        [7, 8, 9],
    ])
    expected_mask = torch.tensor([1, 1, 0, 0], dtype=torch.bool)

    # When
    actual_mask = utils._is_generated_critique_correct(
        actual_critique_ids=actual_critique_ids,
        expected_critique_ids=expected_critique_ids,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Then
    assert torch.all(actual_mask == expected_mask)


def test_get_loss_for_single_correct_example_has_zero_loss(tokenizer):
    # Given
    pad_token_id = tokenizer.pad_token_id
    input_ids = torch.tensor([
        [0, 2, 1, pad_token_id],
    ])
    logits = torch.tensor([
        [
            [0, 0,   0, 0],
            [0, 999, 0, 0],
            [0, 0,   0, 0],
            [0, 0,   0, 0],
        ],
    ], dtype=torch.float32)
    prompt_mask = torch.tensor([
        [1, 1, 0, 0],
    ], dtype=torch.bool)
    expected_loss = torch.tensor(0.0)

    # When
    actual_loss = utils.get_loss(
        input_ids=input_ids,
        logits=logits,
        prompt_mask=prompt_mask,
        pad_token_id=pad_token_id,
    )

    # Then
    assert actual_loss == expected_loss


def test_get_loss_for_single_example(tokenizer):
    # Given
    pad_token_id = tokenizer.pad_token_id
    input_ids = torch.tensor([
        [0, 1, pad_token_id, pad_token_id],
    ])
    logits = torch.tensor([
        [
            [0.1, 0.2, 0.6, 0.1],
            [0.1, 0.2, 0.6, 0.1],
            [0.1, 0.2, 0.6, 0.1],
            [0.1, 0.2, 0.6, 0.1],
        ],
    ])
    prompt_mask = torch.tensor([
        [1, 0, 0, 0],
    ], dtype=torch.bool)
    expected_loss = torch.nn.functional.cross_entropy(
        input=torch.tensor([0.1, 0.2, 0.6, 0.1]),
        target=torch.tensor(1),
    )
    
    # When
    actual_loss = utils.get_loss(
        input_ids=input_ids,
        logits=logits,
        prompt_mask=prompt_mask,
        pad_token_id=pad_token_id,
    )

    # Then
    assert actual_loss == expected_loss


def test_get_loss_for_single_example_with_leading_and_trailing_padding(tokenizer):
    # Given
    pad_token_id = tokenizer.pad_token_id
    input_ids = torch.tensor([
        [pad_token_id, pad_token_id, 1, 0, 1, pad_token_id, pad_token_id],
    ])
    logits = torch.tensor([
        [
            [0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.],
            [2., 0., 0., 0., 0., 0.],
            [0., 3., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
        ],
    ])
    prompt_mask = torch.tensor([
        [1, 1, 1, 0, 0, 0, 0],
    ], dtype=torch.bool)
    expected_loss = torch.nn.functional.cross_entropy(
        input=torch.tensor([
            [2., 0., 0., 0., 0., 0.],
            [0., 3., 0., 0., 0., 0.],
        ]),
        target=torch.tensor([0, 1]),
    )
    
    # When
    actual_loss = utils.get_loss(
        input_ids=input_ids,
        logits=logits,
        prompt_mask=prompt_mask,
        pad_token_id=pad_token_id,
    )

    # Then
    assert actual_loss == expected_loss
