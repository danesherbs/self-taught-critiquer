"""
Finetunes GPT to critique arithmetic problems.
"""

# %%

import datasets
import torch
import transformers
import wandb
import utils

# %%

DEFAULT_CONFIG = {
    "model": "gpt2-medium",
    "manual_seed": 0,
    "device": "cuda:0",
    "few_shot_examples": {
        "n_examples": 3,
        "min_n_digits": 1,
        "max_n_digits": 1,
        "include_rationale_in_critique": True,
        "random_seed": 0,
    },
    "train": {
        "hyperparams": {
            "epochs": 10,
            "lr": 1e-5,
        },
        "dataset": {
            "n_examples": 1_000,
            "min_n_digits": 1,
            "max_n_digits": 1,
            "include_rationale_in_critique": True,
        },
        "dataloader": {
            "batch_size": 8,
            "shuffle": True,
        }
    },
    "test": {
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

def train(**wandb_init_kwargs):
    assert "config" in wandb_init_kwargs, "Must pass config as a kwarg to train()"

    wandb.init(**wandb_init_kwargs)  # start experiment
    
    wandb.define_metric("test_loss", summary="min")
    wandb.define_metric("avg_test_loss", summary="min")
    wandb.define_metric("train_loss", summary="min")
    wandb.define_metric("%_test_accuracy", summary="max")
    
    config = wandb.config  # get config from wandb
    
    torch.manual_seed(config["manual_seed"])

    model = transformers.AutoModelForCausalLM.from_pretrained(config["model"]).to(config["device"])
    model.config.pad_token_id = model.config.eos_token_id

    tokenizer = transformers.AutoTokenizer.from_pretrained(config["model"])
    tokenizer.pad_token = tokenizer.eos_token

    few_shot_examples = datasets.generate_few_shot_examples(**config["few_shot_examples"])
    
    train_data = datasets.ArithmeticDataset(**config["train"]["dataset"])
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        collate_fn=lambda batch: utils.train_collate_fn(
            batch=batch,
            tokenizer=tokenizer,
            few_shot_examples=few_shot_examples,
            device=config["device"],
        ),
        **config["train"]["dataloader"],
    )

    test_data = datasets.ArithmeticDataset(**config["test"]["dataset"])
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        collate_fn=lambda batch: utils.test_collate_fn(
                    batch=batch,
                    tokenizer=tokenizer,
                    few_shot_examples=few_shot_examples,
                    device=config["device"],
        ),
        **config["test"]["dataloader"],
    )

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["train"]["hyperparams"]["lr"],
    )
    
    utils.eval_step(model, tokenizer, test_loader)
    
    for _ in range(config["train"]["hyperparams"]["epochs"]):
        utils.finetune_step(model, optimizer, train_loader, tokenizer.pad_token_id)
        utils.eval_step(model, tokenizer, test_loader)

    wandb.finish()
