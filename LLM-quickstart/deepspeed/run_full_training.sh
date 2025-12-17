#!/bin/bash
# 完整训练流程：T5-3B 和 T5-11B，自动监控和截图

set -e  # 遇到错误立即退出

echo "=========================================="
echo "DeepSpeed ZeRO-3 完整训练流程"
echo "=========================================="
echo ""

# 创建输出目录
mkdir -p training_monitor
mkdir -p training_outputs

# 初始化 CSV 文件
echo "timestamp,model,stage,gpu_name,memory_total_mib,memory_used_mib,memory_free_mib,gpu_util_percent,mem_util_percent,temp_c,power_w" > training_monitor/gpu_metrics.csv

# 函数：抓取快照
capture_snapshot() {
    local model=$1
    local stage=$2
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    
    echo "[$timestamp] 抓取 $model - $stage 快照..."
    
    # nvidia-smi
    nvidia-smi > "training_monitor/nvidia-smi_${model}_${stage}_${timestamp}.txt" 2>&1
    
    # 系统信息
    {
        echo "=== 模型: $model ==="
        echo "=== 阶段: $stage ==="
        echo "=== 时间: $(date) ==="
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
    } > "training_monitor/system-info_${model}_${stage}_${timestamp}.txt"
    
    # 追加到 CSV
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits | \
    sed "s/^/$(date '+%Y-%m-%d %H:%M:%S'),${model},${stage},/" >> training_monitor/gpu_metrics.csv
    
    echo "✅ 已保存快照"
}

# 函数：后台监控
start_monitoring() {
    local model=$1
    echo "启动后台监控（每5分钟一次）..."
    (
        while true; do
            sleep 300  # 5分钟
            capture_snapshot "$model" "monitoring"
        done
    ) &
    MONITOR_PID=$!
    echo "监控进程 PID: $MONITOR_PID"
}

# 函数：停止监控
stop_monitoring() {
    if [ ! -z "$MONITOR_PID" ]; then
        echo "停止监控进程..."
        kill $MONITOR_PID 2>/dev/null || true
    fi
}

# 训练 T5-3B
echo "=========================================="
echo "开始训练 T5-3B 模型"
echo "=========================================="
echo ""

# 训练前快照
capture_snapshot "t5-3b" "before_training"

# 启动后台监控
start_monitoring "t5-3b"

# 训练开始快照（等待几秒让模型加载）
echo "等待模型加载..."
sleep 10
capture_snapshot "t5-3b" "training_start"

# 开始训练
echo "开始训练 T5-3B..."
bash train_t5_3b_zero3.sh > training_outputs/t5-3b_training.log 2>&1 &
TRAIN_PID=$!

# 训练中定期快照（每5分钟）
echo "训练进行中，每5分钟抓取一次快照..."
for i in {1..10}; do  # 最多等待50分钟
    sleep 300
    if ! kill -0 $TRAIN_PID 2>/dev/null; then
        echo "训练已完成"
        break
    fi
    capture_snapshot "t5-3b" "training_${i}"
done

# 等待训练完成
wait $TRAIN_PID || true

# 训练后快照
capture_snapshot "t5-3b" "after_training"

# 停止监控
stop_monitoring

echo ""
echo "=========================================="
echo "T5-3B 训练完成"
echo "=========================================="
echo ""

# 等待一下，清理 GPU 内存
sleep 30

# 训练 T5-11B
echo "=========================================="
echo "开始训练 T5-11B 模型"
echo "=========================================="
echo ""

# 训练前快照
capture_snapshot "t5-11b" "before_training"

# 启动后台监控
start_monitoring "t5-11b"

# 训练开始快照
echo "等待模型加载..."
sleep 15
capture_snapshot "t5-11b" "training_start"

# 开始训练
echo "开始训练 T5-11B..."
bash train_t5_11b_zero3.sh > training_outputs/t5-11b_training.log 2>&1 &
TRAIN_PID=$!

# 训练中定期快照（每5分钟）
echo "训练进行中，每5分钟抓取一次快照..."
for i in {1..15}; do  # 最多等待75分钟
    sleep 300
    if ! kill -0 $TRAIN_PID 2>/dev/null; then
        echo "训练已完成"
        break
    fi
    capture_snapshot "t5-11b" "training_${i}"
done

# 等待训练完成
wait $TRAIN_PID || true

# 训练后快照
capture_snapshot "t5-11b" "after_training"

# 停止监控
stop_monitoring

echo ""
echo "=========================================="
echo "所有训练完成！"
echo "=========================================="
echo ""
echo "输出文件位置:"
echo "  - training_monitor/ (所有截图)"
echo "  - training_outputs/ (训练日志)"
echo "  - output_dir/ (训练结果)"
echo ""

