#!/usr/bin/env python3
"""
ChatGLM3-6B QLoRA 微调前后效果对比脚本

本脚本用于对比微调前后的 ChatGLM3-6B 模型在 AdvertiseGen 数据集上的生成效果。
可以在 Cursor 中直接运行，也可以在 Docker 容器中运行。
"""

import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel, PeftConfig
import pandas as pd
from pathlib import Path


def load_models():
    """加载基础模型和微调后的模型"""
    model_name_or_path = 'THUDM/chatglm3-6b'
    
    print("=" * 80)
    print("正在加载模型...")
    print("=" * 80)
    
    # 加载基础模型
    print("\n1. 加载基础模型...")
    base_model = AutoModel.from_pretrained(
        model_name_or_path,
        device_map='auto',
        trust_remote_code=True
    )
    base_model.requires_grad_(False)
    base_model.eval()
    print("✅ 基础模型加载完成")
    
    # 加载 Tokenizer
    print("\n2. 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    print("✅ Tokenizer 加载完成")
    
    # 加载微调后的模型
    print("\n3. 加载微调后的模型...")
    peft_model_path = "outputs/chatglm3-qlora"
    config = PeftConfig.from_pretrained(peft_model_path)
    fine_tuned_model = PeftModel.from_pretrained(base_model, peft_model_path)
    fine_tuned_model.eval()
    print("✅ 微调后的模型加载完成")
    print(f"训练配置: LoRA r={config.r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    
    return base_model, fine_tuned_model, tokenizer


def compare_responses(query, base_model, fine_tuned_model, tokenizer):
    """对比基础模型和微调后模型的生成结果"""
    device = next(base_model.parameters()).device
    
    # 基础模型生成（使用 generate 方法，更兼容）
    try:
        # 尝试使用 chat 方法
        base_response, _ = base_model.chat(tokenizer, query=query, history=[])
    except AttributeError:
        # 如果 chat 方法失败，使用 generate 方法
        inputs = tokenizer(query, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = base_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.8,
                top_p=0.8,
                pad_token_id=tokenizer.pad_token_id,
            )
        base_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 移除输入部分，只保留生成的部分
        base_response = base_response.replace(query, "").strip()
    
    # 微调后模型生成
    try:
        fine_tuned_response, _ = fine_tuned_model.chat(tokenizer, query=query, history=[])
    except AttributeError:
        # 如果 chat 方法失败，使用 generate 方法
        inputs = tokenizer(query, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = fine_tuned_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.8,
                top_p=0.8,
                pad_token_id=tokenizer.pad_token_id,
            )
        fine_tuned_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 移除输入部分，只保留生成的部分
        fine_tuned_response = fine_tuned_response.replace(query, "").strip()
    
    return base_response, fine_tuned_response


def main():
    """主函数"""
    print("=" * 80)
    print("ChatGLM3-6B QLoRA 微调前后效果对比")
    print("=" * 80)
    
    # 加载模型
    base_model, fine_tuned_model, tokenizer = load_models()
    
    # 测试提示词（来自 AdvertiseGen 数据集）
    test_prompts = [
        "写一段 30 字的广告文案，推广智能理财服务",
        "为一家精品酒店撰写一条促销推送通知",
        "以亲切的口吻，写一个体育用品品牌的引导购买对话"
    ]
    
    print("\n" + "=" * 80)
    print("微调前后效果对比")
    print("=" * 80)
    
    # 实时对比测试
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*80}")
        print(f"测试样本 {i}")
        print(f"{'='*80}")
        print(f"\n📝 Prompt（提示词）:")
        print(f"   {prompt}")
        
        # 获取对比结果
        base_response, fine_tuned_response = compare_responses(
            prompt, base_model, fine_tuned_model, tokenizer
        )
        
        print(f"\n🔵 微调前输出:")
        print(f"   {base_response}")
        
        print(f"\n🟢 微调后输出:")
        # 清理输出文本（移除特殊标记）
        cleaned_response = fine_tuned_response.replace('[gMASK]', '').replace('sop', '').strip()
        print(f"   {cleaned_response}")
        print()
    
    # 显示训练时保存的对比结果
    print("\n" + "=" * 80)
    print("训练时保存的微调前后对比结果")
    print("=" * 80)
    
    comparison_file = Path("outputs/chatglm3-qlora/prompt_comparison.csv")
    if comparison_file.exists():
        comparison_df = pd.read_csv(comparison_file)
        print(f"\n✅ 读取了 {len(comparison_df)} 个测试样本\n")
        
        for idx, row in comparison_df.iterrows():
            print(f"{'='*80}")
            print(f"样本 {idx + 1}")
            print(f"{'='*80}")
            print(f"\n📝 Prompt（提示词）:")
            print(f"   {row['prompt']}")
            
            print(f"\n🔵 微调前输出:")
            if row['pretraining_response'] == '[baseline skipped]':
                print("   [基线测试已跳过]")
            else:
                pre_text = str(row['pretraining_response']).replace('[gMASK]', '').replace('sop', '').strip()
                print(f"   {pre_text[:300]}..." if len(pre_text) > 300 else f"   {pre_text}")
            
            print(f"\n🟢 微调后输出:")
            post_text = str(row['posttraining_response']).replace('[gMASK]', '').replace('sop', '').strip()
            # 移除重复内容
            if post_text.count('"') > 4:
                lines = post_text.split('\n')
                unique_lines = []
                seen = set()
                for line in lines:
                    line_clean = line.strip().strip('"').strip()
                    if line_clean and line_clean not in seen:
                        unique_lines.append(line_clean)
                        seen.add(line_clean)
                post_text = '\n'.join(unique_lines)
            
            print(f"   {post_text[:500]}..." if len(post_text) > 500 else f"   {post_text}")
            print()
    else:
        print(f"\n⚠️  未找到对比文件: {comparison_file}")
    
    print("\n" + "=" * 80)
    print("对比完成！")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

