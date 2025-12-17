# DeepSpeed ZeRO-3 训练作业提交指南

## ✅ 训练状态

训练已自动启动，包括：
- ✅ T5-3B 模型训练（约 15-35 分钟）
- ✅ T5-11B 模型训练（约 40-70 分钟）
- ✅ 自动监控（每5分钟截图一次）

## 📊 监控训练进度

### 方法1: 查看训练日志
```bash
docker exec -it deepspeed-t5-training tail -f training_outputs/t5-3b_training.log
```

### 方法2: 查看完整训练流程日志
```bash
docker exec -it deepspeed-t5-training tail -f full_training.log
```

### 方法3: 检查 GPU 使用情况
```bash
docker exec -it deepspeed-t5-training nvidia-smi
```

### 方法4: 使用 PowerShell 脚本（Windows）
```powershell
.\check_training_status.ps1
```

## 📸 截图文件位置

所有截图自动保存到容器内的 `training_monitor/` 目录：
- `nvidia-smi_*.txt` - GPU 状态截图
- `system-info_*.txt` - 系统信息截图（类似 htop）
- `gpu_metrics.csv` - 所有时间点的 GPU 指标汇总

## 🚀 训练完成后提交步骤

### 1. 准备提交文件（在容器内执行）
```bash
docker exec -it deepspeed-t5-training bash
cd /app
bash prepare_github_submission.sh
```

这将：
- 选择关键时间点的快照
- 创建 README 说明文件
- 准备所有文件到 `github_submission/` 目录

### 2. 复制文件到 Windows（如果需要）
文件已经通过 Docker volume 映射，可以直接在 Windows 中访问：
```
C:\Users\rchua\Desktop\AIFullStackDevelopment\advertisegen_chatglm3_qlora\LLM-quickstart\deepspeed\github_submission\
```

### 3. 提交到 GitHub（在 Windows PowerShell 中）
```powershell
cd C:\Users\rchua\Desktop\AIFullStackDevelopment\advertisegen_chatglm3_qlora

# 添加文件
git add LLM-quickstart/deepspeed/github_submission/*
git add LLM-quickstart/deepspeed/config/ds_config_zero3.json
git add LLM-quickstart/deepspeed/translation/run_translation.py
git add LLM-quickstart/deepspeed/*.sh
git add LLM-quickstart/deepspeed/*.md

# 提交
git commit -m "完成 DeepSpeed ZeRO-3 训练：支持 T5-3B 和 T5-11B 模型训练"

# 推送到 GitHub
git push origin main
```

### 4. 在作业系统中提交

1. 访问 GitHub 仓库
2. 找到 `LLM-quickstart/deepspeed/github_submission/` 目录
3. 选择几个关键时间点的截图文件
4. 在作业系统中提交这些文件的链接

## 📋 推荐提交的截图

### T5-3B 模型
- `nvidia-smi_t5-3b_training_start_*.txt` - 训练开始时的 GPU 状态
- `system-info_t5-3b_training_start_*.txt` - 训练开始时的系统信息
- `nvidia-smi_t5-3b_training_*.txt` - 训练中的 GPU 状态（选择一个）
- `nvidia-smi_t5-3b_after_training_*.txt` - 训练完成后的状态

### T5-11B 模型
- `nvidia-smi_t5-11b_training_start_*.txt` - 训练开始时的 GPU 状态
- `system-info_t5-11b_training_start_*.txt` - 训练开始时的系统信息
- `nvidia-smi_t5-11b_training_*.txt` - 训练中的 GPU 状态（选择一个）
- `nvidia-smi_t5-11b_after_training_*.txt` - 训练完成后的状态

## ⏱️ 预计时间线

- **T5-3B 训练**: 约 15-35 分钟
- **T5-11B 训练**: 约 40-70 分钟
- **总计**: 约 1-2 小时

## 🔧 故障排除

### 如果训练中断
```bash
# 检查容器状态
docker ps -a | grep deepspeed-t5-training

# 重新启动容器（如果已停止）
docker start deepspeed-t5-training

# 重新进入容器
docker exec -it deepspeed-t5-training bash
```

### 如果监控脚本停止
```bash
# 重新启动监控
docker exec -d deepspeed-t5-training bash -c "cd /app && nohup bash monitor_training_simple.sh > monitor.log 2>&1 &"
```

### 查看所有进程
```bash
docker exec -it deepspeed-t5-training bash -c "ps aux | grep -E '(train|deepspeed|monitor)' | grep -v grep"
```

## 📝 注意事项

1. 训练过程中不要关闭 Docker Desktop
2. 确保有足够的磁盘空间（模型和日志文件可能较大）
3. 训练完成后，`github_submission/` 目录会自动准备好提交文件
4. 所有截图都是文本格式（.txt），可以直接在 GitHub 上查看

