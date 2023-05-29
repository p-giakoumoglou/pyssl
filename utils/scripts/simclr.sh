#!/bin/bash

python main.py --model_name=simclr --backbone=resnet18 --batch_size=256 --optimizer=sgd --weight_decay=0.000001 --momentum=0.9 --stop_at_epoch=100 --warmup_epochs=10 --warmup_lr=0 --base_lr=0.3 --final_lr=0 --num_epochs=800

python main_linear.py --model_name=simclr --backbone=resnet18 --batch_size=256 --optimizer=sgd --weight_decay=0 --momentum=0.9 --warmup_epochs=0 --warmup_lr=0 --base_lr=30 --final_lr=0 --num_epochs=30
