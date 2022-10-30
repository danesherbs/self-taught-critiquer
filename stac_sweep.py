import datetime
import stac


config = stac.DEFAULT_CONFIG

name = "stac"
group = f"{name}-{datetime.datetime.now().strftime('%d-%m-%Y-%H:%M:%S')}"

for manual_seed in range(1, 3):
    for batch_size, n_few_shot_examples in [(32, 1), (16, 2), (16, 3), (8, 4), (8, 5), (4, 6), (2, 7), (1, 8), (1, 9)]:
        config["manual_seed"] = manual_seed
        config["train"]["dataloader"]["batch_size"] = batch_size
        config["few_shot_examples"]["n_examples"] = n_few_shot_examples
        stac.train(
            project="seri-mats",
            entity="danesherbs",
            config=config,
            group=group,
        )
