#!/bin/bash
# DeepSpeed ZeRO-3 模式单 GPU 训练 T5-3B 模型
# 使用优化后的配置文件，解决 CPU 内存分配问题

deepspeed --num_gpus=1 translation/run_translation.py \
--deepspeed config/ds_config_zero3.json \
--model_name_or_path t5-3b \
--do_train \
--do_eval \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 4 \
--output_dir output_dir \
--overwrite_output_dir \
--max_train_samples 500 \
--num_train_epochs 1 \
--dataset_name wmt16 \
--dataset_config "ro-en" \
--source_lang en \
--target_lang ro \
--fp16

