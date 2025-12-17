#!/bin/bash
# 准备提交到 GitHub 的文件

echo "=== 准备 GitHub 提交文件 ==="
echo ""

# 创建提交目录
mkdir -p github_submission

echo "1. 选择关键快照..."

# helper: copy first match of a glob (if any)
copy_first_match () {
    local pattern="$1"
    local dest="$2"
    local first
    first=$(ls -1 $pattern 2>/dev/null | head -n 1 || true)
    if [ -n "$first" ]; then
        cp "$first" "$dest/" 2>/dev/null || true
    fi
}

# helper: copy all matches of a glob (if any)
copy_all_matches () {
    local pattern="$1"
    local dest="$2"
    ls -1 $pattern 2>/dev/null | while read -r f; do
        [ -n "$f" ] && cp "$f" "$dest/" 2>/dev/null || true
    done
}

# T5-3B 关键快照：training_start（取一个）
copy_first_match "training_monitor/nvidia-smi_t5-3b_training_start_*.txt" "github_submission"
copy_first_match "training_monitor/system-info_t5-3b_training_start_*.txt" "github_submission"

# 选择训练中的快照（取一个）
T5_3B_MID=$(ls -1 training_monitor/nvidia-smi_t5-3b_training_*.txt 2>/dev/null | head -n 1 || true)
if [ -n "$T5_3B_MID" ]; then
    cp "$T5_3B_MID" github_submission/ 2>/dev/null || true
    cp "${T5_3B_MID/nvidia-smi/system-info}" github_submission/ 2>/dev/null || true
fi

# T5-3B 训练后（取一个）
copy_first_match "training_monitor/nvidia-smi_t5-3b_after_training_*.txt" "github_submission"
copy_first_match "training_monitor/system-info_t5-3b_after_training_*.txt" "github_submission"

# T5-11B 关键快照：training_start（取一个）
copy_first_match "training_monitor/nvidia-smi_t5-11b_training_start_*.txt" "github_submission"
copy_first_match "training_monitor/system-info_t5-11b_training_start_*.txt" "github_submission"

# T5-11B 训练中（取一个）
T5_11B_MID=$(ls -1 training_monitor/nvidia-smi_t5-11b_training_*.txt 2>/dev/null | head -n 1 || true)
if [ -n "$T5_11B_MID" ]; then
    cp "$T5_11B_MID" github_submission/ 2>/dev/null || true
    cp "${T5_11B_MID/nvidia-smi/system-info}" github_submission/ 2>/dev/null || true
fi

# T5-11B 训练后（取一个）
copy_first_match "training_monitor/nvidia-smi_t5-11b_after_training_*.txt" "github_submission"
copy_first_match "training_monitor/system-info_t5-11b_after_training_*.txt" "github_submission"

# 2. 复制 GPU 指标 CSV
if [ -f training_monitor/gpu_metrics.csv ]; then
    cp training_monitor/gpu_metrics.csv github_submission/
fi

# 3. 创建 README 说明文件
cat > github_submission/README.md << 'EOF'
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

EOF

echo "✅ 已准备提交文件到 github_submission/ 目录"
echo ""
echo "文件列表:"
ls -lh github_submission/
echo ""
echo "下一步：将这些文件添加到 Git 并提交到 GitHub"

