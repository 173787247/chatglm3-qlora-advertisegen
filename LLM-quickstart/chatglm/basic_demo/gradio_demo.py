#!/usr/bin/env python3
"""
Gradio WebUI：加载 ChatGLM3-6B 基座模型，并可选加载 LoRA/PEFT adapter（微调产物）。

作业要求点：
- basic_demo 原本不读取 adapter：本 demo 支持通过 peft 加载 `outputs/chatglm3-qlora` 下的 adapter。
- 支持对比：同一条 prompt，分别用「基础模型」与「加载 adapter 的微调模型」生成结果。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModel, AutoTokenizer


DEFAULT_BASE_MODEL = os.environ.get("BASE_MODEL", "THUDM/chatglm3-6b")
DEFAULT_ADAPTER_DIR = os.environ.get("ADAPTER_DIR", str(Path("outputs/chatglm3-qlora")))


@dataclass(frozen=True)
class GenConfig:
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.8


def _device_of(model) -> torch.device:
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


@lru_cache(maxsize=2)
def load_base_model_and_tokenizer(model_name_or_path: str) -> Tuple[object, object]:
    """只加载一次，避免每次交互都重新加载大模型。"""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


@lru_cache(maxsize=2)
def load_finetuned_model(model_name_or_path: str, adapter_dir: str) -> object:
    """在 base model 基础上挂载 LoRA adapter。"""
    base_model, _ = load_base_model_and_tokenizer(model_name_or_path)
    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(f"找不到 adapter 目录：{adapter_path.resolve()}")

    # 仅用于打印/校验配置存在
    _ = PeftConfig.from_pretrained(str(adapter_path))

    tuned = PeftModel.from_pretrained(base_model, str(adapter_path))
    tuned.eval()
    return tuned


def _chat(model, tokenizer, query: str, gen: GenConfig) -> str:
    """兼容 chatglm3 的 .chat；若失败则 fallback 到 generate。"""
    query = (query or "").strip()
    if not query:
        return ""

    try:
        resp, _ = model.chat(
            tokenizer,
            query=query,
            history=[],
            max_new_tokens=gen.max_new_tokens,
            top_p=gen.top_p,
            temperature=gen.temperature,
        )
        return str(resp).strip()
    except Exception:
        device = _device_of(model)
        inputs = tokenizer(query, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen.max_new_tokens,
                do_sample=True,
                temperature=gen.temperature,
                top_p=gen.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text.replace(query, "").strip()


def compare_generate(
    prompt: str,
    base_model_name: str,
    adapter_dir: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, str, str]:
    gen = GenConfig(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)

    base_model, tokenizer = load_base_model_and_tokenizer(base_model_name)
    base_out = _chat(base_model, tokenizer, prompt, gen)

    tuned_out = ""
    status = "✅ 已完成：基础模型 vs 微调模型（LoRA adapter）"
    try:
        tuned_model = load_finetuned_model(base_model_name, adapter_dir)
        tuned_out = _chat(tuned_model, tokenizer, prompt, gen)
    except Exception as e:
        status = f"⚠️ 微调模型加载/推理失败：{e}"

    return base_out, tuned_out, status


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ChatGLM3 LoRA 微调 WebUI（对比）") as demo:
        gr.Markdown(
            "### ChatGLM3-6B LoRA 微调推理对比（基础模型 vs 微调模型）\n"
            "- **基础模型**：直接加载 `THUDM/chatglm3-6b`\n"
            "- **微调模型**：在基础模型上加载 `outputs/chatglm3-qlora` 的 LoRA adapter\n"
        )

        with gr.Row():
            base_model_name = gr.Textbox(
                label="基础模型（HuggingFace ID 或本地路径）",
                value=DEFAULT_BASE_MODEL,
            )
            adapter_dir = gr.Textbox(
                label="LoRA / PEFT Adapter 目录（默认 outputs/chatglm3-qlora）",
                value=DEFAULT_ADAPTER_DIR,
            )

        prompt = gr.Textbox(
            label="请输入 prompt（广告文案/推广语等）",
            lines=4,
            value="写一段 30 字的广告文案，推广智能理财服务",
        )

        with gr.Row():
            max_new_tokens = gr.Slider(32, 1024, value=256, step=32, label="max_new_tokens")
            temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="top_p")

        run_btn = gr.Button("生成对比", variant="primary")

        with gr.Row():
            base_out = gr.Textbox(label="基础模型输出", lines=10)
            tuned_out = gr.Textbox(label="微调模型输出（已加载 adapter）", lines=10)

        status = gr.Markdown()

        run_btn.click(
            fn=compare_generate,
            inputs=[prompt, base_model_name, adapter_dir, max_new_tokens, temperature, top_p],
            outputs=[base_out, tuned_out, status],
        )

        gr.Markdown(
            "### 运行方式\n"
            "在项目根目录 `advertisegen_chatglm3_qlora/` 下执行：\n\n"
            "```bash\n"
            "python LLM-quickstart/chatglm/basic_demo/gradio_demo.py\n"
            "```\n\n"
            "默认会在本机启动 WebUI。Docker 环境下可设置 `--server_name 0.0.0.0`（见下方代码）。"
        )

    return demo


def main():
    demo = build_ui()
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=False,
    )


if __name__ == "__main__":
    main()


