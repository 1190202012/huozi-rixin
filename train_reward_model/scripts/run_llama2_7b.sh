#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
current_time=$(date +%Y-%m-%d_%T)
model=llama2_7b_chat
OUTPUT="./log/$model/$current_time"
mkdir -p $OUTPUT

deepspeed --master_port 14654 --num_gpus 6 main.py \
   --data_output_path ~/tmp/data_files \
   --data_path en_reward_model_dataset \
   --data_split 0,10,0 \
   --model_name_or_path daryl149/llama-2-7b-chat-hf \
   --per_device_train_batch_size 6 \
   --per_device_eval_batch_size 8 \
   --max_seq_len 1024 \
   --learning_rate 9.65e-6 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 3  \
   --gradient_accumulation_steps 4 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 40 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
