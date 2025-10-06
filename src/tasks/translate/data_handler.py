import numpy as np
import os
from datasets import load_dataset


def load_and_prepare_dataset(
        dataset_name,
        tokenizer,
        dataset_config_name,
        train_sample_size,
        eval_sample_size,
        max_source_length,
        max_target_length
):
    """
    加载并预处理适用于中英双向翻译的数据集。

    Args:
        dataset_name (str): Hugging Face Hub上的数据集名称。
        tokenizer: Hugging Face Tokenizer实例。
        dataset_config_name (str): 数据集的特定配置名称（例如语言对）。
        train_sample_size (int): 训练集采样大小。
        eval_sample_size (int): 评估集采样大小。
        max_source_length (int): 输入序列的最大长度。
        max_target_length (int): 输出序列的最大长度。

    Returns:
        tuple: 包含处理后的训练集、评估集和None（因为S2S任务没有num_labels）。
    """
    print(f"🔄 正在从 Hub 加载数据集: {dataset_name}, 配置: {dataset_config_name}")
    # 1. 加载数据集
    raw_datasets = load_dataset(dataset_name, name=dataset_config_name)

    # 2. 定义任务前缀，这是让模型知道翻译方向的关键
    prefix_zh_to_en = "将中文翻译为英文: "
    prefix_en_to_zh = "translate English to Chinese: "

    def preprocess_function(examples):
        """对数据进行预处理，为双向翻译创建样本"""
        inputs = []
        targets = []

        # examples['translation'] 是一个list of dict, 每个dict是 {'en': '...', 'zh': '...'}
        for ex in examples["translation"]:
            # 跳过可能存在的空数据
            if not ex['en'] or not ex['zh']:
                continue

            # 创建 中文 -> 英文 的样本
            inputs.append(prefix_zh_to_en + ex["zh"].strip())
            targets.append(ex["en"].strip())

            # 创建 英文 -> 中文 的样本
            inputs.append(prefix_en_to_zh + ex["en"].strip())
            targets.append(ex["zh"].strip())

        # 对输入文本进行编码
        model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True)

        # 使用 target_tokenizer 上下文管理器对目标文本进行编码
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # 3. 先对原始数据进行采样
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]  # 通常使用validation集做评估

    if train_sample_size:
        print(f"🔪 对训练集进行采样，保留 {train_sample_size} 条样本...")
        # 为了保证多样性，随机采样
        train_dataset = train_dataset.shuffle(seed=42).select(range(train_sample_size))

    if eval_sample_size:
        print(f"🔪 对评估集进行采样，保留 {eval_sample_size} 条样本...")
        eval_dataset = eval_dataset.shuffle(seed=42).select(range(eval_sample_size))

    print("⚙️  正在对采样后的数据集进行预处理...")
    # 获取CPU核心数用于并行处理
    num_proc = os.cpu_count()
    print(f"🔧 使用 {num_proc} 个CPU核心进行并行处理...")
    
    # 使用 .map() 方法高效地应用预处理函数到采样后的数据
    # remove_columns 会删除原始的'translation'列，因为我们已经处理完并生成了'input_ids', 'labels'等
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=raw_datasets["train"].column_names
    )
    
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=raw_datasets["validation"].column_names
    )

    print(f"✅ 数据准备完成。训练集大小: {len(train_dataset)}, 评估集大小: {len(eval_dataset)}")

    # Seq2Seq 任务不返回 num_labels，所以返回 None
    return train_dataset, eval_dataset, None
