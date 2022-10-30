"""
Teaches GPT to critique arithmetic problems by finetuning on its own outputs.
"""

# %%

from typing import Dict, Any
import datasets
import torch
import transformers
import wandb
import utils

# %%

DEFAULT_CONFIG = {
    "model": "gpt2-medium",
    "device": "cuda:0",
    "manual_seed": 0,
    "n_stac_iters": 4,
    "few_shot_examples": {
        "n_examples": 3,
        "min_n_digits": 1,
        "max_n_digits": 1,
        "include_rationale_in_critique": True,
        "random_seed": 0,
    },
    "generate": {
        "few_shot_discriminator": {
            "enabled": False,
            "n_examples": 5,
        },
        "dataset": {
            "n_examples": 2 ** 11,
            "min_n_digits": 1,
            "max_n_digits": 1,
            "include_rationale_in_critique": True,
        },
        "dataloader": {
            "batch_size": 64,
            "shuffle": True,
        }
    },
    "train": {
        "hyperparams": {
            "epochs": 5,
            "lr": 1e-5,
            "include_few_shot_examples": True,
        },
        "dataloader": {
            "batch_size": 16,
            "shuffle": True,
        }
    },
    "test": {
        "hyperparams": {
            "include_few_shot_examples": True,
        },
        "dataset": {
            "n_examples": 2 ** 8,
            "min_n_digits": 1,
            "max_n_digits": 1,
            "include_rationale_in_critique": True,
        },
        "dataloader": {
            "batch_size": 8,
            "shuffle": True,
        }
    }
}

# %%

def _make_pretrained_model(model_name: str):
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = model.config.eos_token_id
    return model

# %%

def _make_generate_dataloader(tokenizer: Any, few_shot_examples: str, config: Dict[str, Any]):
    generate_data = datasets.ArithmeticDataset(**config["generate"]["dataset"])
    generate_loader = torch.utils.data.DataLoader(
        dataset=generate_data,
        collate_fn=lambda batch: utils.generate_collate_fn(
            batch=batch,
            tokenizer=tokenizer,
            few_shot_examples=few_shot_examples,
            device=config["device"],
        ),
        **config["generate"]["dataloader"],
    )
    return generate_loader

# %%

def _make_train_dataloader(tokenizer, few_shot_examples, mask, config):
    train_few_shot_examples = few_shot_examples
        
    if not config["train"]["hyperparams"]["include_few_shot_examples"]:
        train_few_shot_examples = ""

    train_data = datasets.ArithmeticDataset(
        mask=mask,
        **config["generate"]["dataset"],
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        collate_fn=lambda batch: utils.train_collate_fn(
            batch=batch,
            tokenizer=tokenizer,
            few_shot_examples=train_few_shot_examples,
            device=config["device"],
        ),
        **config["train"]["dataloader"],
    )

    return train_loader

# %%

def _make_test_dataloader(tokenizer, few_shot_examples, config):
    test_few_shot_examples = few_shot_examples
    
    if not config["test"]["hyperparams"]["include_few_shot_examples"]:
        test_few_shot_examples = ""

    test_data = datasets.ArithmeticDataset(**config["test"]["dataset"])
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        collate_fn=lambda batch: utils.test_collate_fn(
            batch=batch,
            tokenizer=tokenizer,
            few_shot_examples=test_few_shot_examples,
            device=config["device"],
        ),
        **config["test"]["dataloader"],
    )

    return test_loader

# %%

def _generate_step(model, tokenizer, dataloader, enable_few_shot_discriminator, n_few_shot_examples):
    model.eval()
    masks = []
    
    for prompts, expected_critiques in dataloader:
        gen_tensors = utils.generate_critiques(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            expected_critiques=expected_critiques,
            enable_logging=True,
            pad_token_id=tokenizer.pad_token_id,
            return_intermediate_tensors=True,
            enable_few_shot_discriminator=enable_few_shot_discriminator,
            n_few_shot_discriminator_examples=n_few_shot_examples,
        )

        mask = gen_tensors["mask"]
        masks.append(gen_tensors["mask"])

        wandb.log({"correct_critiques_in_batch": int(torch.sum(mask))})
    
    mask = torch.cat(masks, dim=0)
    
    wandb.log({"correct_critiques_in_generate_step": int(torch.sum(mask))})
    
    return mask

# %%

def train(**wandb_init_kwargs):
    assert "config" in wandb_init_kwargs, "Must pass config as a kwarg to train()"
    
    wandb.init(**wandb_init_kwargs)

    wandb.define_metric("test_loss", summary="min")
    wandb.define_metric("avg_test_loss", summary="min")
    wandb.define_metric("train_loss", summary="min")
    wandb.define_metric("%_test_accuracy", summary="max")

    config = wandb.config

    torch.manual_seed(config["manual_seed"])

    tokenizer = transformers.AutoTokenizer.from_pretrained(config["model"])
    tokenizer.pad_token = tokenizer.eos_token

    few_shot_examples = datasets.generate_few_shot_examples(**config["few_shot_examples"])
    generate_loader = _make_generate_dataloader(tokenizer, few_shot_examples, config)
    test_loader = _make_test_dataloader(tokenizer, few_shot_examples, config)

    models = [_make_pretrained_model(config["model"]) if n == 0 else None for n in range(config["n_stac_iters"]+1)]
    models[0].to(config["device"])

    for n in range(1, config["n_stac_iters"]+1):
        mask = _generate_step(
            model=models[n-1],
            tokenizer=tokenizer,
            dataloader=generate_loader,
            enable_few_shot_discriminator=config["generate"]["few_shot_discriminator"]["enabled"],
            n_few_shot_examples=config["generate"]["few_shot_discriminator"]["n_examples"],
        )
        models[n-1].to("cpu")

        models[n] = _make_pretrained_model(config["model"])
        models[n].to(config["device"])

        optimizer = torch.optim.Adam(
            lr=config["train"]["hyperparams"]["lr"],
            params=models[n].parameters(),
        )

        train_loader = _make_train_dataloader(tokenizer, few_shot_examples, mask, config)

        utils.eval_step(
            model=models[n],
            tokenizer=tokenizer,
            dataloader=test_loader,
        )

        for _ in range(config["train"]["hyperparams"]["epochs"]):
            utils.finetune_step(
                model=models[n],
                optimizer=optimizer,
                dataloader=train_loader,
                pad_token_id=tokenizer.pad_token_id,
            )

            utils.eval_step(
                model=models[n],
                tokenizer=tokenizer,
                dataloader=test_loader,
            )

        # keep model on GPU for next iteration
    
    wandb.finish()
# %%
