# tasks/story_generation/data_handler.py

from datasets import load_dataset


def load_and_prepare_dataset(dataset_name, tokenizer, train_sample_size, eval_sample_size, max_length,
                             dataset_config_name=None, **kwargs):
    """
    为 Causal LM 任务加载并准备数据集。

    Args:
        dataset_name (str): 数据集名称。
        tokenizer: Hugging Face Tokenizer 实例。
        train_sample_size (int): 训练集采样大小。
        eval_sample_size (int): 评估集采样大小。
        max_length (int): 分词时的最大长度。
        dataset_config_name (str, optional): 数据集的特定配置。
        **kwargs: 接收 main.py 传入的其他多余参数，以保持接口兼容性。

    Returns:
        tuple: (tokenized_train_dataset, tokenized_eval_dataset, num_labels)
               对于CausalLM任务, num_labels 为 None。
    """
    print(f"🔄 正在加载数据集 '{dataset_name}'...")
    # 加载完整数据集，包含训练集和验证集
    full_dataset = load_dataset(dataset_name, name=dataset_config_name)
    train_dataset = full_dataset['train']
    eval_dataset = full_dataset['validation']

    # 如果指定了采样大小，则对数据集进行采样
    if train_sample_size:
        print(f"🔍 采样 {train_sample_size} 条数据作为训练集...")
        train_dataset = train_dataset.select(range(train_sample_size))
    
    if eval_sample_size:
        print(f"🔍 采样 {eval_sample_size} 条数据作为评估集...")
        eval_dataset = eval_dataset.select(range(eval_sample_size))

    print(f"训练集大小: {len(train_dataset)}, 评估集大小: {len(eval_dataset)}")

    def tokenize_function(examples):
        """分词函数，处理 'story' 字段"""
        # 对文本进行分词，并截断到 max_length
        # 注意：这里不进行填充，让 DataCollator 在批处理时统一处理
        output = tokenizer(
            examples["story"],
            truncation=True,
            max_length=max_length,
            padding=False,  # 不在这里填充，让 DataCollator 处理
            return_special_tokens_mask=False,  # 不需要特殊token掩码
        )
        # 对于 CausalLM 任务, labels 就是 input_ids 的一个副本
        # Trainer 会自动处理向右移位以进行下一个token预测
        output["labels"] = output["input_ids"].copy()
        return output

    print("🧠 正在对数据集进行分词...")
    tokenized_train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    tokenized_eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    print("✅ 数据集准备完成！")
    # Causal LM 任务没有 num_labels 的概念，返回 None
    return tokenized_train_dataset, tokenized_eval_dataset, None