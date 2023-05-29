#!/bin/bash

python main.py --model_name=simsiam --backbone=resnet18 --batch_size=512 --optimizer=sgd --weight_decay=0.0005 --momentum=0.9 --stop_at_epoch=800 --warmup_epochs=10 --warmup_lr=0 --base_lr=0.03 --final_lr=0 --num_epochs=800

python main_linear.py --model_name=simsiam --backbone=resnet18 --batch_size=256 --optimizer=sgd --weight_decay=0 --momentum=0.9 --warmup_epochs=10 --warmup_lr=0 --base_lr=30 --final_lr=0 --num_epochs=100
