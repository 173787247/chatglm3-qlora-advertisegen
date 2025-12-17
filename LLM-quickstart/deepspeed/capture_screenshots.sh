#!/bin/bash
# 截图脚本：保存 nvidia-smi 和系统信息用于作业提交

echo "=== 捕获系统信息用于作业提交 ==="
echo ""

# 创建输出目录
mkdir -p screenshots

# 1. 保存 nvidia-smi 输出
echo "1. 保存 nvidia-smi 输出..."
nvidia-smi > screenshots/nvidia-smi.txt 2>&1
echo "✅ 已保存: screenshots/nvidia-smi.txt"

# 2. 保存系统信息（替代 htop）
echo "2. 保存系统信息..."
{
    echo "=== CPU 信息 ==="
    echo "CPU 核心数: $(nproc)"
    echo ""
    echo "=== 内存信息 ==="
    free -h
    echo ""
    echo "=== 进程信息（Top 10 by CPU）==="
    ps aux --sort=-%cpu | head -11
    echo ""
    echo "=== 进程信息（Top 10 by Memory）==="
    ps aux --sort=-%mem | head -11
} > screenshots/system-info.txt
echo "✅ 已保存: screenshots/system-info.txt"

# 3. 保存详细的 nvidia-smi 信息
echo "3. 保存详细 GPU 信息..."
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv > screenshots/nvidia-smi-detail.csv 2>&1
echo "✅ 已保存: screenshots/nvidia-smi-detail.csv"

echo ""
echo "=== 完成 ==="
echo "所有信息已保存到 screenshots/ 目录"
echo ""
echo "文件列表:"
ls -lh screenshots/

