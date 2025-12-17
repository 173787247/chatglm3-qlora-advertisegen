#!/bin/bash
# 训练监控脚本：每5分钟自动抓取 nvidia-smi 和系统信息

echo "=== DeepSpeed 训练监控脚本 ==="
echo "每5分钟自动抓取一次 nvidia-smi 和系统信息"
echo "按 Ctrl+C 停止监控"
echo ""

# 创建输出目录
mkdir -p training_monitor

# 计数器
counter=1

# 无限循环，每5分钟抓取一次
while true; do
    timestamp=$(date '+%Y%m%d_%H%M%S')
    echo "[$counter] $(date '+%Y-%m-%d %H:%M:%S') - 抓取训练状态..."
    
    # 1. 保存 nvidia-smi 输出
    nvidia-smi > "training_monitor/nvidia-smi_${timestamp}.txt" 2>&1
    
    # 2. 保存系统信息
    {
        echo "=== 时间戳: $(date) ==="
        echo ""
        echo "=== CPU 信息 ==="
        echo "CPU 核心数: $(nproc)"
        echo ""
        echo "=== 内存信息 ==="
        free -h
        echo ""
        echo "=== GPU 详细信息 ==="
        nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv
        echo ""
        echo "=== Top 10 进程（按 CPU）==="
        ps aux --sort=-%cpu | head -11
        echo ""
        echo "=== Top 10 进程（按内存）==="
        ps aux --sort=-%mem | head -11
        echo ""
        echo "=== 训练相关进程 ==="
        ps aux | grep -E '(train|deepspeed|python.*translation)' | grep -v grep || echo "未找到训练进程"
    } > "training_monitor/system-info_${timestamp}.txt"
    
    # 3. 保存简化的 CSV 格式（便于后续分析）
    {
        echo "timestamp,gpu_name,memory_total_mib,memory_used_mib,memory_free_mib,gpu_util_percent,mem_util_percent,temp_c,power_w"
        nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits | sed "s/^/$(date '+%Y-%m-%d %H:%M:%S'),/"
    } >> "training_monitor/gpu_metrics.csv"
    
    echo "✅ 已保存: training_monitor/nvidia-smi_${timestamp}.txt"
    echo "✅ 已保存: training_monitor/system-info_${timestamp}.txt"
    echo "✅ 已追加: training_monitor/gpu_metrics.csv"
    echo ""
    
    counter=$((counter + 1))
    
    # 等待5分钟（300秒）
    echo "等待5分钟..."
    sleep 300
done

