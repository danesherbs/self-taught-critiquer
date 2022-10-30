import datetime
import stac


config = stac.DEFAULT_CONFIG
config["generate"]["few_shot_discriminator"]["enabled"] = True
config["few_shot_examples"]["n_examples"] = 4

name = "stac-discriminator"
group = f"{name}-{datetime.datetime.now().strftime('%d-%m-%Y-%H:%M:%S')}"

for manual_seed in range(3):
    for n_discriminator_few_shot_examples in range(10):
        config["manual_seed"] = manual_seed
        config["generate"]["few_shot_discriminator"]["n_examples"] = n_discriminator_few_shot_examples
        stac.train(
            project="seri-mats",
            entity="danesherbs",
            config=config,
            group=group,
        )
