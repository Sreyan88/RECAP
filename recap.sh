#!/bin/bash

python train.py

python infer.py --model_path /path_to_saved_ckpts --checkpoint_path checkpoint_name/
