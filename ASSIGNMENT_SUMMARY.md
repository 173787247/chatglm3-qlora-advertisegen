# 作业完成总结

## 第二次作业完成情况

### 要求1: 基于 data 目录下的数据训练 ChatGLM3 模型，使用 inference Notebook 对比微调前后的效果

**✅ 已完成**

#### 1. 训练部分
- **训练脚本**: `train_chatglm3_advertise.py`
  - 使用 `--dataset_path data/advertisegen.json` 参数加载数据
  - 支持从 `data` 目录读取 JSON 格式的训练数据
  - 已完成 3 epochs 训练，Loss 从 4.23 降至 2.27

- **数据集**: `data/advertisegen.json`
  - 包含 10,000 个 AdvertiseGen 样本
  - 格式：`instruction` 和 `response` 字段

- **训练结果**:
  - 模型文件: `outputs/chatglm3-qlora/adapter_model.safetensors`
  - Loss 曲线: `outputs/chatglm3-qlora/loss_curve.csv`
  - 训练日志: `outputs/chatglm3-qlora/training_log_history.csv`

#### 2. Inference 对比部分
- **Inference Notebook**: `chatglm3_inference_comparison.ipynb`
  - 加载基础模型和微调后的模型
  - 实现微调前后效果对比功能
  - 包含测试样本对比和训练时保存的对比结果

- **Inference 脚本**: `inference_comparison.py`
  - 可在命令行直接运行
  - 支持在 Docker 容器中运行（与训练环境一致）

#### 3. 对比结果
- 训练时自动保存对比结果到 `outputs/chatglm3-qlora/prompt_comparison.csv`
- Inference Notebook 可以实时对比任意测试样本
- 展示了微调前后模型生成质量的明显提升

**提交文件**:
- ✅ `train_chatglm3_advertise.py`
- ✅ `data/advertisegen.json`
- ✅ `chatglm3_inference_comparison.ipynb`
- ✅ `inference_comparison.py`
- ✅ `outputs/chatglm3-qlora/` (训练结果目录)

---

### 要求2: 将 gen_dataset Notebook 改写为 py 文件

**✅ 已完成**

- **原文件**: `LLM-quickstart/chatglm/gen_dataset.ipynb`
- **转换后**: `gen_dataset.py`
  - 216 行代码
  - 包含完整的数据生成功能
  - 支持使用 OpenAI GPT-4o-mini 和 LangChain 生成训练数据集

**提交文件**:
- ✅ `gen_dataset.py`

---

## 第一次作业完成情况

### 要求: 使用 QLoRA 在 ChatGLM3-6B 上微调，并观察 Loss 变化情况，对比微调前后模型生成内容质量

**✅ 已完成**

- 训练完成，Loss 曲线已记录
- 微调前后对比已完成
- 所有结果文件已提交

---

## 文件提交状态

所有文件已成功提交到 GitHub:
- 仓库地址: https://github.com/173787247/chatglm3-qlora-advertisegen
- 两次作业文件互不影响，完整保留

---

## 运行说明

### 训练模型
```bash
python train_chatglm3_advertise.py --dataset_path data/advertisegen.json --output_dir outputs/chatglm3-qlora --num_train_epochs 3
```

### 运行 Inference 对比
```bash
# 方式1: 使用 Notebook
jupyter notebook chatglm3_inference_comparison.ipynb

# 方式2: 使用 Python 脚本
python inference_comparison.py

# 方式3: 在 Docker 中运行
docker exec -it chatglm3-qlora-training python /app/inference_comparison.py
```

### 生成数据集（gen_dataset.py）
```bash
python gen_dataset.py
```

