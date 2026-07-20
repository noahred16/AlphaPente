#!/bin/bash

# hf auth login
hf download noahred16/alpha-pente-model best_model.pt --local-dir ./checkpoints/pente/
hf download noahred16/alpha-pente-data bootstrap.pt --repo-type dataset --local-dir ./checkpoints/pente/


# model repo
# hf upload noahred16/alpha-pente-model ./checkpoints/pente/best_model.pt best_model.pt

# dataset repo (needs --repo-type dataset, just like the download did)
# hf upload noahred16/alpha-pente-data ./checkpoints/pente/bootstrap.pt bootstrap.pt --repo-type dataset
