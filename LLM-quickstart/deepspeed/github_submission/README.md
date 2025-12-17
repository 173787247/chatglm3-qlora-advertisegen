# DeepSpeed ZeRO-3 训练截图

## 训练配置

- **模型**: T5-3B 和 T5-11B
- **优化方法**: DeepSpeed ZeRO-3
- **GPU**: NVIDIA GeForce RTX 5080 (16GB)
- **训练样本**: 500 个
- **训练轮数**: 1 epoch

## 文件说明

### T5-3B 模型
- `nvidia-smi_t5-3b_training_start_*.txt` - 训练开始时的 GPU 状态
- `system-info_t5-3b_training_start_*.txt` - 训练开始时的系统信息
- `nvidia-smi_t5-3b_training_*.txt` - 训练中的 GPU 状态
- `system-info_t5-3b_training_*.txt` - 训练中的系统信息
- `nvidia-smi_t5-3b_after_training_*.txt` - 训练完成后的 GPU 状态
- `system-info_t5-3b_after_training_*.txt` - 训练完成后的系统信息

### T5-11B 模型
- `nvidia-smi_t5-11b_training_start_*.txt` - 训练开始时的 GPU 状态
- `system-info_t5-11b_training_start_*.txt` - 训练开始时的系统信息
- `nvidia-smi_t5-11b_training_*.txt` - 训练中的 GPU 状态
- `system-info_t5-11b_training_*.txt` - 训练中的系统信息
- `nvidia-smi_t5-11b_after_training_*.txt` - 训练完成后的 GPU 状态
- `system-info_t5-11b_after_training_*.txt` - 训练完成后的系统信息

### 汇总数据
- `gpu_metrics.csv` - 所有时间点的 GPU 指标汇总（CSV 格式）

## 提交说明

这些截图展示了：
1. ZeRO-3 配置成功支持 T5-3B 和 T5-11B 模型训练
2. GPU 和 CPU 资源使用情况
3. 训练过程中的资源变化

