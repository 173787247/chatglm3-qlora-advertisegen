# basic_demo（已适配 LoRA Adapter）

本目录用于完成作业要求：**让 demo 支持读取微调后的 LoRA/Adapter，并通过 WebUI 调用**。

## 关键文件

- `gradio_demo.py`：Gradio WebUI，对比「基础模型」与「加载 LoRA adapter 的微调模型」输出。

## 运行

在 `advertisegen_chatglm3_qlora/` 目录下执行：

```bash
python LLM-quickstart/chatglm/basic_demo/gradio_demo.py
```

可选环境变量：

- `BASE_MODEL`：基础模型（默认 `THUDM/chatglm3-6b`）
- `ADAPTER_DIR`：LoRA adapter 目录（默认 `outputs/chatglm3-qlora`）
- `GRADIO_SERVER_NAME`：服务监听地址（Docker 里建议 `0.0.0.0`）
- `GRADIO_SERVER_PORT`：端口（默认 `7860`）

## 作业截图建议

- WebUI 页面中同一条 prompt 的两列输出（基础 vs 微调）
- 终端启动日志（能看到 adapter 路径与模型加载）


