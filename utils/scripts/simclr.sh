#!/bin/bash

python main.py --model_name=simclr --backbone=resnet50 --batch_size=512 --optimizer=lars_simclr --weight_decay=0.000001 --momentum=0.9 --stop_at_epoch=100 --warmup_epochs=10 --warmup_lr=0 --base_lr=0.5 --final_lr=0.5 --num_epochs=100

python main_linear.py --model_name=simclr --backbone=resnet50 --batch_size=512 --optimizer=sgd_nesterov --weight_decay=0 --momentum=0.9 --warmup_epochs=0 --warmup_lr=0 --base_lr=0.1 --final_lr=0.1 --num_epochs=300
