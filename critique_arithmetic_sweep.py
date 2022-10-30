import datetime
import critique_arithmetic


config = critique_arithmetic.DEFAULT_CONFIG

name = "critique_arithmetic"
group = f"{name}-{datetime.datetime.now().strftime('%d-%m-%Y-%H:%M:%S')}"

for manual_seed in range(3):
    for n_examples in [2 ** n for n in range(4, 13)]:
        config["manual_seed"] = manual_seed
        config["train"]["dataset"]["n_examples"] = n_examples
        critique_arithmetic.train(
            project="seri-mats",
            entity="danesherbs",
            config=config,
            group=group,
        )
