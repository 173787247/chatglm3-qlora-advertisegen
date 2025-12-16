# pyright: reportMissingImports=false
"""
QLoRA 微调 ChatGLM3-6B（AdvertiseGen）的小脚本，支持 Loss 记录和微调前后对比。
推荐 4-bit + LoRA 在 RTX-5080/256GB 的环境中执行，最终可将 notebook/脚本上传 GitHub。
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

DEFAULT_PROMPTS = [
    "写一段 30 字的广告文案，推广智能理财服务",
    "为一家精品酒店撰写一条促销推送通知",
    "以亲切的口吻，写一个体育用品品牌的引导购买对话",
]


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA 微调 ChatGLM3-6B on AdvertiseGen")
    parser.add_argument("--model_name", type=str, default="THUDM/chatglm3-6b")
    parser.add_argument("--output_dir", type=str, default="outputs/chatglm3-qlora")
    parser.add_argument("--num_samples", type=int, default=10_000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dataset_path", type=str, default=None, help="本地 JSON/JSONL 文件或目录，格式包含 instruction/response")
    parser.add_argument("--prompt_template", type=str, default="问题：{instruction}\n回答：{response}", help="格式化每条样本的模板，需包含 {instruction} 和 {response}")
    parser.add_argument("--max_input_length", type=int, default=512, help="最多保留多少字符的 instruction")
    parser.add_argument("--max_output_length", type=int, default=512, help="最多保留多少字符的 response")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--bnb_dtype", type=str, default="float16")
    parser.add_argument("--bnb_mode", type=str, choices=["4bit", "8bit", "none"], default="4bit", help="BitsAndBytes 量化模式（none=不使用量化）")
    parser.add_argument("--bnb_double_quant", type=str, default="True", help="是否启用 double quantization（True/False）")
    parser.add_argument("--bnb_llm_int8_threshold", type=float, default=6.0, help="LLM int8 threshold，4bit/8bit 模式都可调整")
    parser.add_argument("--train_dtype", type=str, choices=["fp16", "bf16", "float32"], default="fp16")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--lora_r", type=int, default=4, help="LoRA rank (r) 值")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha 值")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout 率")
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["query_key_value"],
        help="LoRA 插件要注入的模块名列表（参考 ChatGLM 结构）",
    )
    parser.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="跳过微调前基线文本生成，直接进入 LoRA + Trainer。",
    )
    return parser.parse_args()


def str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"无法解析布尔值：{value}")


def dtype_from_name(name: str) -> torch.dtype:
    if not hasattr(torch, name):
        raise ValueError(f"torch has no dtype {name}")
    return getattr(torch, name)


def prepare_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer


def prepare_dataset(
    tokenizer: AutoTokenizer,
    num_samples: int,
    max_length: int,
    dataset_path: str | None,
    prompt_template: str,
    max_input_length: int,
    max_output_length: int,
):
    if dataset_path:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    else:
        dataset = load_dataset("AdvertiseGen", split="train")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    instruction_key = "instruction" if "instruction" in dataset.column_names else "content"
    response_key = "response" if "response" in dataset.column_names else "summary"

    def build_text(example):
        instruction = str(example.get(instruction_key, ""))[:max_input_length]
        response = str(example.get(response_key, ""))[:max_output_length]
        return {"text": prompt_template.format(instruction=instruction, response=response)}

    dataset = dataset.map(build_text, remove_columns=dataset.column_names)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch")
    return tokenized


def ensure_hidden_layers(config):
    if not hasattr(config, "num_hidden_layers"):
        candidate = getattr(config, "num_layers", None)
        if candidate is not None:
            setattr(config, "num_hidden_layers", candidate)
    return config


def load_quantized_model(model_name: str, bnb_config: BitsAndBytesConfig = None):
    if bnb_config is None:
        # 不使用量化，直接加载完整模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        # 使用量化配置
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
        )
    ensure_hidden_layers(model.config)
    ensure_past_extractor(model)
    model.config.use_cache = False
    return model


def ensure_past_extractor(model: AutoModelForCausalLM):
    if not hasattr(model, "_extract_past_from_model_output"):
        def _extract_past_from_model_output(self, outputs):
            past = getattr(outputs, "past_key_values", None)
            return (None, past)
        setattr(model.__class__, "_extract_past_from_model_output", _extract_past_from_model_output)


def attach_lora(model: AutoModelForCausalLM, args) -> AutoModelForCausalLM:
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def build_trainer(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, args):
    use_fp16 = args.train_dtype == "fp16"
    use_bf16 = args.train_dtype == "bf16"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0 if os.name == 'nt' else 4,  # Windows 使用单进程，Linux/Docker 使用多进程
        dataloader_pin_memory=torch.cuda.is_available(),  # 仅在 CUDA 可用时启用 pin_memory
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    progress_callback = TrainingProgressCallback(logging.getLogger(__name__))
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=args.train_dataset,
        data_collator=collator,
        callbacks=[progress_callback],
    )
class TrainingProgressCallback(TrainerCallback):
    """打印每隔 logging_steps 的进度，避免 Trainer 卡住看不到输出。"""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not logs:
            return
        step = logs.get("step")
        loss = logs.get("loss")
        if step is not None and loss is not None:
            self._logger.info("step=%s loss=%s elapsed=%s", step, loss, state.log_history[-1].get("train_runtime", ""))



def _resolve_device(model: AutoModelForCausalLM) -> torch.device:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Iterable[str],
    max_new_tokens: int,
    temperature: float,
) -> List[str]:
    model.eval()
    results = []
    device = _resolve_device(model)
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append(text)
    return results


def export_log_history(trainer: Trainer, output_dir: str):
    history = pd.DataFrame(trainer.state.log_history)
    history_path = Path(output_dir) / "training_log_history.csv"
    history.to_csv(history_path, index=False)
    LOGGER.info("Log history saved to %s", history_path)
    return history


def export_comparisons(prompts: List[str], before: List[str], after: List[str], output_dir: str):
    data = []
    for prompt, base_response, fine_response in zip(prompts, before, after):
        data.append(
            {
                "prompt": prompt,
                "pretraining_response": base_response,
                "posttraining_response": fine_response,
            }
        )
    df = pd.DataFrame(data)
    comparison_path = Path(output_dir) / "prompt_comparison.csv"
    df.to_csv(comparison_path, index=False)
    LOGGER.info("Prompt comparison saved to %s", comparison_path)
    return df


def main():
    args = parse_args()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    if not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        raise SystemExit(
            "请先在 advertisegen_chatglm3_qlora/.env 中设置 "
            "HUGGINGFACE_HUB_TOKEN，或者在环境变量里设置。"
        )

    LOGGER.info("Loading tokenizer and dataset...")
    tokenizer = prepare_tokenizer(args.model_name)
    args.train_dataset = prepare_dataset(
        tokenizer,
        args.num_samples,
        args.max_length,
        args.dataset_path,
        args.prompt_template,
        args.max_input_length,
        args.max_output_length,
    )
    LOGGER.info("Dataset ready: %s samples", len(args.train_dataset))

    # 配置量化（如果启用）
    if args.bnb_mode == "none":
        LOGGER.info("不使用量化，直接加载完整模型")
        bnb_config = None
    else:
        double_quant = str_to_bool(args.bnb_double_quant)
        bnb_dtype = dtype_from_name(args.bnb_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.bnb_mode == "4bit",
            load_in_8bit=args.bnb_mode == "8bit",
            bnb_4bit_compute_dtype=bnb_dtype,
            bnb_4bit_use_double_quant=double_quant,
            bnb_8bit_compute_dtype=bnb_dtype,
            llm_int8_threshold=args.bnb_llm_int8_threshold,
        )

    LOGGER.info("Loading base model for pre-training comparison...")
    base_model = load_quantized_model(args.model_name, bnb_config)
    if args.skip_baseline:
        LOGGER.info("跳过微调前基线生成")
        base_responses = ["[baseline skipped]" for _ in args.prompts]
    else:
        base_responses = generate_responses(
            base_model,
            tokenizer,
            args.prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    LOGGER.info("Attaching LoRA adapters...")
    model = attach_lora(base_model, args)
    trainer = build_trainer(model, tokenizer, args)

    LOGGER.info("Starting training for %s epochs ...", args.num_train_epochs)
    LOGGER.info("训练数据集大小: %d 样本", len(args.train_dataset))
    # 计算总步数
    num_samples = len(args.train_dataset)
    batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    steps_per_epoch = (num_samples + batch_size - 1) // batch_size  # 向上取整
    total_steps = steps_per_epoch * args.num_train_epochs
    LOGGER.info("预计总步数: %d (每 epoch %d 步)", total_steps, steps_per_epoch)
    LOGGER.info("每 %d 步输出一次日志，请耐心等待...", args.logging_steps)
    try:
        trainer.train()
    except Exception as e:
        LOGGER.error("训练出错: %s", e, exc_info=True)
        raise
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    LOGGER.info("Recording training logs...")
    history = export_log_history(trainer, args.output_dir)

    LOGGER.info("Generating post-training samples...")
    post_responses = generate_responses(
        model,
        tokenizer,
        args.prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    export_comparisons(args.prompts, base_responses, post_responses, args.output_dir)

    loss_df = history.loc[history["loss"].notna(), ["step", "loss"]]
    loss_path = output_path / "loss_curve.csv"
    loss_df.to_csv(loss_path, index=False)
    LOGGER.info("Loss steps saved to %s", loss_path)

    LOGGER.info("训练完成，输出目录：%s", args.output_dir)


if __name__ == "__main__":
    main()

