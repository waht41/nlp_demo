# tasks/story_generation/data_handler.py

from datasets import load_dataset


def load_and_prepare_dataset(dataset_name, tokenizer, train_sample_size, eval_sample_size, max_length,
                             dataset_config_name=None, **kwargs):
    """
    为 "Prompt-to-Story" Causal LM 任务加载并准备数据集。

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
    full_dataset = load_dataset(dataset_name, name=dataset_config_name)
    train_dataset = full_dataset['train']
    eval_dataset = full_dataset['validation']

    if train_sample_size:
        print(f"🔍 采样 {train_sample_size} 条数据作为训练集...")
        train_dataset = train_dataset.select(range(train_sample_size))

    if eval_sample_size:
        print(f"🔍 采样 {eval_sample_size} 条数据作为评估集...")
        eval_dataset = eval_dataset.select(range(eval_sample_size))

    print(f"训练集大小: {len(train_dataset)}, 评估集大小: {len(eval_dataset)}")

    def tokenize_function(examples):
        """
        ### 核心改动 ###
        目标: 将 'prompt' 和 'story' 拼接为一条完整的序列。
        方法: 1. 拼接文本: prompt + eos_token + story + eos_token
              2. 直接对拼接后的文本进行分词。
        我们不再手动创建 labels，这项工作将完全交给 DataCollatorForLanguageModeling。
        """
        # 1. 将 prompt 和 story 拼接成一个完整的输入文本
        #    eos_token 用于分隔 prompt 和 story，并标记序列结束
        full_texts = [
            p + tokenizer.eos_token + s + tokenizer.eos_token
            for p, s in zip(examples['prompt'], examples['story'])
        ]

        # 2. 对拼接后的文本进行分词
        model_inputs = tokenizer(
            full_texts,
            truncation=True,
            max_length=max_length,
            padding=False,  # 动态填充由 DataCollator 在每个批次中处理，效率更高
        )
        return model_inputs

    print("🧠 ----------------------------------------------------")
    print("🧠 正在以 'Prompt-to-Story' 模式对数据集进行分词...")
    print("🧠 (输入 = prompt + story, 标签由 DataCollator 自动生成)") # 更新了提示信息
    print("🧠 ----------------------------------------------------")

    tokenized_train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="处理训练集..."
    )
    tokenized_eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="处理评估集..."
    )

    print("✅ 数据集准备完成！")
    return tokenized_train_dataset, tokenized_eval_dataset, None