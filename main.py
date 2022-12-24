#!/bin/env python
import wandb

settings = wandb.Settings(enable_job_creation=True)

with wandb.init(settings=settings):
    for i in range(100):
        wandb.log({"metric": i})
    
