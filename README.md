# ChatGLM3-6B + QLoRA on AdvertiseGen

这个目录提供一个 `python` 脚本，可在 RTX-5080/256GB 的环境下微调 ChatGLM3-6B，满足课程要求的 10K 样本训练、Loss 变化记录与微调前后输出比对。

## 1. 环境依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate peft bitsandbytes pandas
pip install python-dotenv
```

建议用 `venv` 或 `conda` 创建独立环境，并确保 Docker Desktop 的 GPU 镜像能访问显卡。

## 2. 数据来源

官方 `AdvertiseGen` 训练集来自 HuggingFace，如果你的环境不能直接访问（如需要登录或该数据在你所在区域不可见），可以手动下载数据并保存为包含 `instruction` 与 `response` 字段的 JSON/JSONL 文件，随后在脚本里通过 `--dataset_path path/to/data.json` 指定。

## 2. 数据准备脚本

脚本会自动读取 `advertisegen_chatglm3_qlora/.env`（需含 `HUGGINGFACE_HUB_TOKEN`）并设置 token，以便直接从 HuggingFace 下载。
新增加了 `download_advertisegen.py`（直接抓取 `shibing624/AdvertiseGen` 的前 10K 并生成 `data/advertisegen.json`）以及 `run_advertisegen_qlora.ps1`，把准备数据和训练串在一起。
如果你手上有包含 instruction/response 的 CSV/JSON/JSONL 文件（例如老师给的原始 AdvertiseGen 数据），先运行：

```bash
python prepare_advertise_data.py \
  --input_paths path/to/raw_dir path/to/other.jsonl \
  --output_path data/advertisegen.json \
  --max_samples 10000
```

该脚本可以读取目录下所有 `.json`, `.jsonl`, `.csv`，将字段映射到 `instruction`/`response` 并生成标准训练文件，供 `train_chatglm3_advertise.py --dataset_path data/advertisegen.json` 使用。

## 3. 运行训练

**说明**：两种方式都可以使用 GPU（RTX-5080）。Windows 上只要安装了 NVIDIA 驱动、CUDA Toolkit 和 CUDA 版本的 PyTorch，就可以直接使用 GPU，不需要 Docker。Docker 主要提供环境隔离和一致性。

### 方式一：Docker Desktop + WSL2（**强烈推荐**，避免 Windows bitsandbytes 问题）

**重要**：Windows 上 `bitsandbytes` 4-bit 量化在训练反向传播时容易卡住（`dequantize_4bit` 操作），建议使用 Docker 运行。

确保 Docker Desktop 已启用 GPU 支持（Settings → Resources → Advanced → 启用 GPU）。

**使用清华大学镜像源（推荐，加速下载）：**

```powershell
# 方式一：使用脚本自动配置并拉取镜像
.\docker-pull-tsinghua.ps1

# 方式二：手动配置 Docker 镜像加速器
# 在 Docker Desktop -> Settings -> Docker Engine 中添加：
# {
#   "registry-mirrors": [
#     "https://docker.mirrors.ustc.edu.cn",
#     "https://hub-mirror.c.163.com"
#   ]
# }
# 然后重启 Docker Desktop

# 构建并运行
docker-compose build
docker-compose up

# 后台运行并查看日志
docker-compose up -d
docker-compose logs -f

# 进入容器手动执行
docker exec -it chatglm3-qlora-training bash
python train_chatglm3_advertise.py --dataset_path data/advertisegen.json ...
```

### 方式二：Windows 本地直接运行（需要 CUDA 版本的 PyTorch）

**前提条件**：
- 已安装 NVIDIA 驱动
- 已安装 CUDA Toolkit（RTX-5080 支持 CUDA 12.x）
- 已安装 CUDA 版本的 PyTorch：
  ```powershell
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

**运行训练**：
```powershell
python train_chatglm3_advertise.py `
  --dataset_path data/advertisegen.json `
  --output_dir outputs/chatglm3-qlora `
  --num_samples 10000 `
  --per_device_train_batch_size 2 `
  --gradient_accumulation_steps 8 `
  --num_train_epochs 3 `
  --bnb_dtype float16 `
  --train_dtype fp16 `
  --prompt_template "问题：{instruction}`n回答：{response}" `
  --bnb_mode 4bit `
  --bnb_double_quant True `
  --bnb_llm_int8_threshold 6.0 `
  --lora_r 4 `
  --lora_alpha 32 `
  --lora_dropout 0.05 `
  --target_modules query_key_value
```

**验证 CUDA 是否可用**：
```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

你也可以借鉴 [LLM-quickstart/peft/peft_qlora_chatglm.ipynb](https://github.com/DjangoPeng/LLM-quickstart/blob/main/peft/peft_qlora_chatglm.ipynb) 提供的逻辑：
- 通过 `--prompt_template` 自定义「问题/回答」格式（默认 `问题：{instruction}\n回答：{response}`），保持与 notebook 中的 prompt 一致。
- 传入 `--lora_r/--lora_alpha/--lora_dropout` 与 `--target_modules` 调整 LoRA 适配器（默认注入 `query_key_value`）。
- 用 `--max_input_length` 与 `--max_output_length` 拟合 notebook 中为 instruction/response 分别设定 max token 的思路，避免过长语料。
- 使用 `--train_dtype` 选择训练数据类型 (`fp16`/`bf16`/`float32`)，并在脚本内部调用 `prepare_model_for_kbit_training` 以匹配 notebook 中的 QLoRA 预处理。
- 如果 4-bit 量化阶段卡住，可以临时加上 `--bnb_mode 8bit`（或 `--bnb_double_quant False`）使用 8-bit 策略；`--bnb_llm_int8_threshold` 也可调节 QLoRA/8-bit 的容错门限。
```

`train_chatglm3_advertise.py` 脚本会：

- 下载并处理 AdvertiseGen 训练集（10K 样本）；
- 使用 BitsAndBytes 4-bit + LoRA 训练 ChatGLM3-6B；
- 在 `outputs/chatglm3-qlora/` 记录模型、Loss CSV、prompt 对比等；
- 留下 micro prompt 的「训练前后」输出，方便放进 notebook/报告。

## 3. 结果整理

运行结束后，目录结构如下：

- `outputs/chatglm3-qlora/`（LoRA 权重 + tokenizer）
- `training_log_history.csv`（包含每个 logging step 的 loss）
- `loss_curve.csv`（提取 `step` + `loss` 方便画图）
- `prompt_comparison.csv`（训练前/后的生成对比）

将上述内容组合进 `.ipynb`：

1. 用 pandas 读 `loss_curve.csv` 画曲线；
2. 显示 `prompt_comparison.csv`；
3. 记录 hardware config（RTX-5080 + 16GB GPU...）；
4. push 生成的 notebook 到 GitHub，复制链接提交。

## 4. 进阶

- 若需要监控 TensorBoard，可在脚本里开启 `report_to="tensorboard"`，然后跑 `tensorboard --logdir outputs/chatglm3-qlora/logs`；
- 想增加 eval，建议在 `Trainer` 中加入 `eval_dataset`，并监听 `eval_loss`。

如需我把这个流程转成 `.ipynb` 模板，或者写脚本自动生成 loss 图、图表截屏，可接着说。

