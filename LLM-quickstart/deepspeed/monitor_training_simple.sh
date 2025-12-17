#!/bin/bash
# 简化版监控脚本：每5分钟抓取一次（后台运行）

# 创建输出目录
mkdir -p training_monitor

# 初始化 CSV 文件（如果不存在）
if [ ! -f "training_monitor/gpu_metrics.csv" ]; then
    echo "timestamp,gpu_name,memory_total_mib,memory_used_mib,memory_free_mib,gpu_util_percent,mem_util_percent,temp_c,power_w" > training_monitor/gpu_metrics.csv
fi

counter=1

while true; do
    timestamp=$(date '+%Y%m%d_%H%M%S')
    
    # 抓取 nvidia-smi
    nvidia-smi > "training_monitor/nvidia-smi_${timestamp}.txt" 2>&1
    
    # 抓取系统信息
    {
        echo "=== $(date) ==="
        free -h
        echo ""
        nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv
        echo ""
        ps aux --sort=-%cpu | head -11
    } > "training_monitor/system-info_${timestamp}.txt"
    
    # 追加到 CSV
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits | sed "s/^/$(date '+%Y-%m-%d %H:%M:%S'),/" >> training_monitor/gpu_metrics.csv
    
    echo "[$counter] $(date '+%H:%M:%S') - 已保存快照 #$counter"
    counter=$((counter + 1))
    sleep 300  # 5分钟
done

