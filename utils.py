import torch
import wandb

from torch import Tensor
from typing import Dict, Sequence, Tuple
from transformers import PreTrainedTokenizerBase


def get_loss(input_ids, logits, prompt_mask, pad_token_id):
    """Compute the loss for a batch of inputs"""
    
    logits = logits[:, :-1, :]  # take one off end [batch_size, seq_len-1, vocab_size]
    labels = input_ids[:, 1:].clone()  # take one off start [batch_size, seq_len-1]
    prompt_mask = prompt_mask[:, 1:]  # take one off start
    labels[prompt_mask] = pad_token_id  # pad labels where prompts are
    
    # reshape to [batch_size * (seq_len-1), vocab_size]
    logits_flattened = logits.reshape(-1, logits.shape[-1])
    
    # reshape to [batch_size * (seq_len-1),]
    labels_flattened = labels.reshape(-1)

    loss = torch.nn.functional.cross_entropy(
        input=logits_flattened,
        target=labels_flattened,
        ignore_index=pad_token_id,
        reduction="mean",
    )
    
    return loss


def generate_critiques(model, tokenizer, prompts, expected_critiques, pad_token_id, enable_logging=False, return_intermediate_tensors=False, enable_few_shot_discriminator=False, n_few_shot_discriminator_examples=-1):
    """Generate critiques for a batch of prompts"""

    assert not enable_few_shot_discriminator, "Not implemented yet."
    assert n_few_shot_discriminator_examples == -1, "Few-shot discriminator not implemented yet."

    prompts_len = prompts["input_ids"].shape[-1]
    critiques_len = expected_critiques["input_ids"].shape[-1]
    
    model_generations = model.generate(
        **prompts,
        max_new_tokens=critiques_len,
        do_sample=False,
    )

    actual_critique_ids = model_generations[:, prompts_len:]
    expected_critique_ids = expected_critiques["input_ids"]

    mask = _is_generated_critique_correct(
        actual_critique_ids=actual_critique_ids,
        expected_critique_ids=expected_critique_ids,
        pad_token_id=pad_token_id,
    )

    correct_critiques = _keep_only_masked_examples(
        prompts=prompts,
        completions=expected_critiques,
        mask=mask,
    )

    if enable_logging:
        decoded_prompts = tokenizer.batch_decode(
            prompts["input_ids"],
            skip_special_tokens=True,
        )

        decoded_actual_critiques = tokenizer.batch_decode(
            actual_critique_ids,
            skip_special_tokens=True,
        )

        decoded_expected_critiques = tokenizer.batch_decode(
            expected_critique_ids,
            skip_special_tokens=True,
        )

        sampled_critiques = wandb.Table(
            data=[[
                    decoded_prompts[i],
                    decoded_actual_critiques[i],
                    decoded_expected_critiques[i],
                    bool(mask[i]),
                ] for i in range(len(decoded_prompts))
            ],
            columns=["prompt", "actual_critiques", "expected_critiques", "is_correct"],
        )

        wandb.log({
            "sampled_critiques": sampled_critiques,
        })

    if return_intermediate_tensors:
        return {
            "correct_critiques": correct_critiques,
            "actual_critique_ids": actual_critique_ids,
            "mask": mask,
        }

    return correct_critiques


def _is_generated_critique_correct(actual_critique_ids: Tensor, expected_critique_ids: Tensor, pad_token_id: int) -> Tensor:
    """Returns a mask of True/False for each example in the batch"""

    _, n = expected_critique_ids.shape
    matches = actual_critique_ids[:, :n] == expected_critique_ids
    pad_token_mask = expected_critique_ids == pad_token_id
    matches[pad_token_mask] = True
    return torch.sum(matches, dim=-1) == n


def _keep_only_masked_examples(prompts: Dict[str, Tensor], completions: Dict[str, Tensor], mask: Tensor) -> Dict[str, Tensor]:
    """Keep only examples where mask is True"""

    filtered_prompts_input_ids = prompts["input_ids"][mask]
    filtered_prompts_attn_mask = prompts["attention_mask"][mask]

    filtered_completions_input_ids = completions["input_ids"][mask]
    filtered_completions_attn_mask = completions["attention_mask"][mask]
    
    filtered_input_ids = torch.cat([filtered_prompts_input_ids, filtered_completions_input_ids], dim=-1)
    filtered_attn_masks = torch.cat([filtered_prompts_attn_mask, filtered_completions_attn_mask], dim=-1)

    prompt_mask_lhs = torch.ones_like(filtered_prompts_input_ids, dtype=torch.bool)
    prompt_mask_rhs = torch.zeros_like(filtered_completions_input_ids, dtype=torch.bool)
    prompt_mask = torch.cat([prompt_mask_lhs, prompt_mask_rhs], dim=-1)

    return {
        "input_ids": filtered_input_ids,
        "attention_mask": filtered_attn_masks,
        "prompt_mask": prompt_mask,
    }


def finetune_step(model, optimizer, dataloader, pad_token_id):
    model.train()
    
    for input_ids, attention_mask, prompt_mask in dataloader:
        """
                                        v
        inputs:     the cat sat on  the mat   (take one off end)
        labels: the cat sat on  the mat       (take one off start)
                ^
        """
        assert input_ids.shape == attention_mask.shape == prompt_mask.shape

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = get_loss(
            input_ids=input_ids,
            logits=logits,
            prompt_mask=prompt_mask,
            pad_token_id=pad_token_id,
        )
        loss.backward()
        optimizer.step()

        wandb.log({"train_loss": float(loss)})


def eval_step(model, tokenizer, dataloader):
    model.eval()
    loss = 0.0
    avg_loss = 0.0
    test_num_correct = 0

    for i, (input_ids, attention_mask, prompt_mask, prompts, expected_critiques) in enumerate(dataloader):
        with torch.no_grad():
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            ).loss
        
        avg_loss += float(loss)
        is_last_batch = i == len(dataloader) - 1

        gen_tensors = generate_critiques(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            expected_critiques=expected_critiques,
            pad_token_id=tokenizer.pad_token_id,
            enable_logging=is_last_batch,
            return_intermediate_tensors=True,
        )
        
        num_correct_in_batch = int(gen_tensors["mask"].sum())
        test_num_correct += num_correct_in_batch

        wandb.log({
            "test_loss": float(loss),
            "correct_critiques_in_test_batch": num_correct_in_batch,
        })

    wandb.log({
        "avg_test_loss": avg_loss / (i+1),
        "%_test_accuracy": 100 * test_num_correct / len(dataloader.dataset),
    })


def generate_collate_fn(batch: Sequence[Tuple[str, str]], tokenizer: PreTrainedTokenizerBase, few_shot_examples: str, device: str) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor], Sequence[str], Sequence[str]]:
    decoded_prompts = [few_shot_examples + prompt for prompt, _ in batch]

    tokenizer.padding_side = "left"  # for batch generation
    prompts = tokenizer(
        decoded_prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    ).to(device)
    tokenizer.padding_side = "right"  # undo
    
    decoded_expected_critiques = [critique for _, critique in batch]

    expected_critiques = tokenizer(
        decoded_expected_critiques,
        padding=True,
        truncation=False,
        return_tensors="pt",
    ).to(device)
    
    return prompts, expected_critiques


def train_collate_fn(batch: Sequence[Tuple[str, str]], tokenizer: PreTrainedTokenizerBase, few_shot_examples: str, device: str) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    examples = [few_shot_examples + prompt + critique for prompt, critique in batch]

    inputs = tokenizer(
        examples,
        padding=True,
        truncation=False,
        return_tensors="pt",
    ).to(device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    prompt_mask = torch.zeros_like(
        input_ids,
        dtype=torch.bool,
        device=device,
    )

    for i, (prompt, _) in enumerate(batch):
        prompt_length = len(tokenizer(few_shot_examples + prompt).input_ids)
        prompt_mask[i, :prompt_length] = True
    
    return input_ids, attention_mask, prompt_mask


def test_collate_fn(batch: Sequence[Tuple[str, str]], tokenizer: PreTrainedTokenizerBase,few_shot_examples: str, device: str):
    prompts, expected_critiques = generate_collate_fn(
        batch=batch,
        tokenizer=tokenizer,
        few_shot_examples=few_shot_examples,
        device=device,
    )

    input_ids = torch.cat([prompts["input_ids"], expected_critiques["input_ids"]], dim=-1).to(device)
    attention_mask = torch.cat([prompts["attention_mask"], expected_critiques["attention_mask"]], dim=-1).to(device)

    prompt_mask = torch.cat([
        torch.ones_like(prompts["input_ids"]),
        torch.zeros_like(expected_critiques["input_ids"]),
    ], dim=-1)

    return input_ids, attention_mask, prompt_mask, prompts, expected_critiques

