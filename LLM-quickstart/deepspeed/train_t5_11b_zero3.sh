#!/bin/bash
# DeepSpeed ZeRO-3 模式单 GPU 训练 T5-11B 模型
# 使用优化后的配置文件，支持更大模型训练

deepspeed --num_gpus=1 translation/run_translation.py \
--deepspeed config/ds_config_zero3.json \
--model_name_or_path google/t5-v1_1-xxl \
--do_train \
--do_eval \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--output_dir output_dir \
--overwrite_output_dir \
--max_train_samples 500 \
--num_train_epochs 1 \
--dataset_name wmt16 \
--dataset_config "ro-en" \
--source_lang en \
--target_lang ro \
--fp16

