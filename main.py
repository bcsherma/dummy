import wandb
import random

config = dict(
    learning_rate=0.01,
    dropout=0.2,
    batch_size=32,
    epochs=10,
    architecture="CNN",
    gradient_accumulation_steps=2,
)

with wandb.init(config=config):
    for i in range(wandb.config.epochs):
        wandb.log({"metric": i * random.random()})
