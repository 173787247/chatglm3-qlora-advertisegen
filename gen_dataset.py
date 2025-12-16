#!/usr/bin/env python3
"""
构造微调训练数据集

借助 ChatGPT 和 GPT API 我们可以实现自动化批量构造训练数据集。

下面我们以中国古典哲学数据集为例，展示了自动构造训练集的主要流程：

- 使用 LangChain 构造训练数据样例
    - 基于 ChatGPT 设计 `System Role` 提示词
    - 使用 `OpenAI GPT-4o-mini` 生成基础数据
    - 解析 OpenAI GPT 生成的训练数据
    - 持久化存储`dataset.csv`训练数据集文件
    - 使用 ChatGPT 实现训练数据多样化
- 自动化批量生成训练数据集
    - 整理收集原始数据`raw_data.txt`
    - 自动解析原始数据样例 `raw_data_content[]`
    - 设计 `gen_data` 训练数据生成器函数
    - 设计训练数据生成流水线

最佳实践参考：
- 使用 GPT-4o-mini 生成基础数据：https://platform.openai.com/playground/p/2c7XNPgo6Y2iDxILiWfD3iPu?model=gpt-4o-mini&mode=chat
- 使用 ChatGPT 生成数据处理代码和相关文本整理：https://chat.openai.com/share/cdfd2d1d-a75e-4cee-be49-539c010ca1b1
- GPT API 价格: https://openai.com/pricing
"""

import csv
import datetime
import os
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


# 初始化 OpenAI 客户端
client = OpenAI()

# 初始化 LangChain 的 GPT-4o-mini 调用
chat = ChatOpenAI(model="gpt-4o-mini", temperature=1, max_tokens=4095)


def gen_data(raw_content):
    """
    使用 LangChain GPT-4o-mini 调用处理单个数据样例。

    :param raw_content: 原始数据样例。
    :return: GPT-4o-mini 模型生成的内容。
    """
    # 系统消息定义背景和任务
    system_message = SystemMessage(
        content="""
        你是中国古典哲学大师，尤其擅长周易的哲学解读。

        接下来，你收到的都是关于周易卦象的解释，你需要整理润色，并生成用于大模型训练的内容和格式。

        示例输入：

        师卦，此卦是异卦相叠，下卦为坎，上卦为坤。"师"指军队。坎为水、为险；坤为地、为顺，喻寓兵于农。兵凶战危，用兵乃圣人不得已而为之，但它可以顺利无阻碍地解决矛盾，因为顺乎形势，师出有名，故能化凶为吉。占得此卦，对于军事上率师出征非常有利，必无灾祸。师卦是天马出群之卦，以寡伏众之象。
        师卦位于讼卦之后，《序卦》之中这样解释道："讼必有众起，故受之以师。师者，众也。"争讼的人越来越多，以致形成了军队。

        期待结果：

        content:"师卦"
        summary:"在周易中，师卦是一个极具深意的卦象，它由两个异卦相叠组成：下卦坎（水）和上卦坤（地）。这一卦象代表"师"，即军队，寓意着兵力和农力的结合。在这里，坎卦象征着水和险难，而坤卦象征着地和顺从，暗示着通过将军事力量安置于民间，可以在必要时顺利调动。

        师卦的核心哲学是：虽然兵力代表着危险和战争，但其使用应当是圣人不得已而为之的最后手段。在正确的情况下，军事力量可以顺应形势，将危险转化为吉祥。因此，在军事策略上，此卦象征着出征将会顺利，无灾祸。

        师卦紧随讼卦（争讼卦），在《序卦》中解释为"讼必有众起，故受之以师"。这意味着争端激化至众多人群的参与，形成了类似军队的集体力量。"

        返回格式要求：
        content:"{卦名}"
        summary:"{内容}"
        """
    )

    # 人类消息包含原始数据样例
    human_message = HumanMessage(content=raw_content)

    # 构建消息列表并进行模型调用
    messages = [system_message, human_message]
    ai_message = chat(messages)

    return ai_message.content


def dataset_parser(ai_message_content):
    """
    解析由 gen_data 函数生成的 ai_message.content，提取 content 和 summary。

    :param ai_message_content: gen_data 函数返回的文本。
    :return: 提取的 content 和 summary。
    """
    # 分割字符串来找到 content 和 summary 的位置
    content_start = ai_message_content.find('content:"') + len('content:"')
    content_end = ai_message_content.find('"\nsummary:')
    summary_start = ai_message_content.find('summary:"') + len('summary:"')
    summary_end = ai_message_content.rfind('"')

    # 提取并存储 content 和 summary
    content = ai_message_content[content_start:content_end].strip()
    summary = ai_message_content[summary_start:summary_end].strip()

    return content, summary


def generate_question_summary_pairs(content, summary):
    """
    生成20对提问和总结的配对。

    :param content: 内容（例如："蒙卦"）。
    :param summary: 内容的总结。
    :return: 包含20对提问和总结的列表。
    """
    # 20种提问模板
    question_templates = [
        "{}代表什么？",
        "周易中的{}含义是什么？",
        "请解释一下{}。",
        "{}在周易中是什么象征？",
        "周易{}的深层含义是什么？",
        "{}和教育启蒙有什么联系？",
        "周易的{}讲述了什么？",
        "{}是怎样的一个卦象？",
        "{}在周易中怎样表达教育的概念？",
        "{}的基本意义是什么？",
        "周易中{}的解释是什么？",
        "{}在周易中代表了哪些方面？",
        "{}涉及哪些哲学思想？",
        "周易中{}的象征意义是什么？",
        "{}的主要讲述内容是什么？",
        "周易{}的核心思想是什么？",
        "{}和启蒙教育之间有何联系？",
        "在周易中，{}象征着什么？",
        "请描述{}的含义。",
        "{}在周易哲学中扮演什么角色？"
    ]

    # 使用 content 填充提问模板
    questions = [template.format(content) for template in question_templates]

    # 创建提问和总结的配对
    question_summary_pairs = [(question, summary) for question in questions]

    return question_summary_pairs


def main():
    """
    主函数：自动化批量生成训练数据集流水线
    
    原始数据来源：https://www.zhouyi.cc/zhouyi/yijing64/4103.html
    """
    # 确保 data 目录存在
    if not os.path.exists('data'):
        os.makedirs('data')

    # 解析 data/raw_data.txt 得到 raw_content_data 列表
    raw_content_data = []
    raw_data_path = 'LLM-quickstart/chatglm/data/raw_data.txt'
    
    if not os.path.exists(raw_data_path):
        print(f"警告: 未找到原始数据文件 {raw_data_path}")
        print("请确保 raw_data.txt 文件存在")
        return
    
    with open(raw_data_path, 'r', encoding='utf-8') as file:
        content = file.read()
        data_samples = content.split('\n\n')
        for sample in data_samples:
            cleaned_sample = sample.strip()
            if cleaned_sample:
                raw_content_data.append(cleaned_sample)

    print(f"读取到 {len(raw_content_data)} 个原始数据样例")

    # 创建带有时间戳的 CSV 文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/zhouyi_dataset_{timestamp}.csv"

    # 创建 CSV 文件并写入标题行
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['content', 'summary'])

        # 循环遍历 raw_content_data 数据样例
        for idx, raw_content in enumerate(raw_content_data, 1):
            print(f"\n处理第 {idx}/{len(raw_content_data)} 个样例...")
            
            try:
                # 调用 gen_data 方法得到 ai_message_content
                ai_message_content = gen_data(raw_content)

                # 解析 ai_message_content 得到 content 和 summary
                content, summary = dataset_parser(ai_message_content)
                
                print(f"Content: {content}")
                print(f"Summary: {summary[:100]}...")  # 只显示前100个字符

                # 调用 generate_question_summary_pairs 得到20组 pairs
                pairs = generate_question_summary_pairs(content, summary)

                # 将 pairs 写入 csv 文件
                for pair in pairs:
                    writer.writerow(pair)
                    
            except Exception as e:
                print(f"处理第 {idx} 个样例时出错: {e}")
                continue

    print(f"\n✅ 数据集生成完成！保存至: {filename}")


if __name__ == "__main__":
    main()

