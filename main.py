#!/bin/env python
import wandb

with wandb.init():
    for i in range(100):
        wandb.log({"metric": i})
    
