from datasets import load_dataset

def load_and_prepare_dataset(dataset_name: str, tokenizer, train_sample_size=None, eval_sample_size=None, **kwargs):
    """
    专门为 SNLI (或类似的句子对NLI任务) 加载和预处理数据。
    处理流程：
    1. 加载数据集
    2. 过滤无效样本
    3. 提取并采样训练集和验证集
    4. 对采样后的数据进行分词处理
    """
    print(f"正在加载数据集: {dataset_name}")
    raw_datasets = load_dataset(dataset_name)

    # 1. **关键步骤**: 过滤掉label为-1的样本
    # 这些样本在SNLI中表示标注者没有达成一致，不应用于训练或评估
    print("正在过滤无效样本 (label == -1)...")
    filtered_datasets = raw_datasets.filter(lambda example: example['label'] != -1)

    # 2. 提取训练集和验证集
    train_dataset = filtered_datasets["train"]
    eval_dataset = filtered_datasets["validation"]

    # 3. 根据配置进行采样
    if train_sample_size:
        print(f"对训练集进行采样，大小: {train_sample_size}")
        train_dataset = train_dataset.select(range(train_sample_size))
    if eval_sample_size:
        print(f"对评估集进行采样，大小: {eval_sample_size}")
        eval_dataset = eval_dataset.select(range(eval_sample_size))

    # 4. 定义预处理函数 (处理句子对)
    def preprocess_function(examples):
        # SNLI 数据集的文本列名为 'premise' 和 'hypothesis'
        # 分词器会自动将它们格式化为：[CLS] premise [SEP] hypothesis [SEP]
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            max_length=128
        )

    print("正在对采样后的数据集进行分词处理...")
    # 对采样后的数据集进行分词处理
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)

    # 5. 推断标签数量
    # 注意：从过滤后的数据集中获取特征
    num_labels = filtered_datasets['train'].features['label'].num_classes

    return train_dataset, eval_dataset, num_labels