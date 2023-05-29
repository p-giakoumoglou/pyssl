#!/bin/bash

python main.py --model_name=simclr --backbone=resnet18 -batch_size=512 --optimizer=lars_simclr --weight_decay=0.0001 --momentum=0.9 --warmup_epochs=10 --warmup_lr=0 --base_lr=1.0 --final_lr=0 --num_epochs=1000

python main_linear.py --model_name=simclr --backbone=resnet18--batch_size=512 --optimizer=sgd_nesterov --weight_decay=0 --momentum=0.9 --warmup_epochs=0 --base_lr=0.1 --final_lr=0 --num_epochs=100