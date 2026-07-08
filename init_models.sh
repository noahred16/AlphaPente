#!/bin/bash

# hf auth login
hf download noahred16/alpha-pente-model best_model.pt --local-dir ./checkpoints/pente/
hf download noahred16/alpha-pente-data bootstrap.pt --repo-type dataset --local-dir ./checkpoints/pente/

