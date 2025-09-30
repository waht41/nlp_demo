# data_handler.py
# 负责加载和预处理数据集

from datasets import load_dataset


def load_and_prepare_dataset(dataset_name, tokenizer, text_column="text", label_column="label", train_sample_size=None,
                             eval_sample_size=None):
    """
    从Hugging Face Hub加载数据集，进行采样和分词处理。

    Args:
        dataset_name (str): 数据集的名称 (例如 "rotten_tomatoes")。
        tokenizer: 使用的transformers分词器实例。
        text_column (str): 数据集中包含文本的列名。
        label_column (str): 数据集中包含标签的列名。
        train_sample_size (int, optional): 训练集采样大小，用于快速测试。默认为None，使用完整数据集。
        eval_sample_size (int, optional): 评估集采样大小，用于快速测试。默认为None，使用完整数据集。

    Returns:
        tuple: (tokenized_train_dataset, tokenized_eval_dataset, num_labels)
    """
    print(f"正在加载数据集 '{dataset_name}'...")
    # 加载数据集
    dataset = load_dataset(dataset_name)

    # 获取标签信息
    labels = dataset["train"].features[label_column].names
    num_labels = len(labels)
    print(f"数据集标签: {labels} (共 {num_labels} 类)")

    # 定义分词函数
    def tokenize_function(examples):
        return tokenizer(examples[text_column], padding="max_length", truncation=True)

    print("正在对数据集进行分词...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 移除不再需要的原始文本列，并重命名标签列
    tokenized_datasets = tokenized_datasets.remove_columns([text_column])
    tokenized_datasets = tokenized_datasets.rename_column(label_column, "labels")
    tokenized_datasets.set_format("torch")

    # 根据需要进行采样
    train_dataset = tokenized_datasets["train"]
    if train_sample_size:
        print(f"对训练集进行采样，大小为: {train_sample_size}")
        train_dataset = train_dataset.shuffle(seed=42).select(range(train_sample_size))

    # Rotten Tomatoes没有验证集，我们从测试集中拆分一个
    if "validation" in tokenized_datasets:
        eval_dataset = tokenized_datasets["validation"]
    else:
        print("数据集中没有验证集，将从测试集中拆分一个。")
        eval_dataset = tokenized_datasets["test"]

    if eval_sample_size:
        print(f"对评估集进行采样，大小为: {eval_sample_size}")
        eval_dataset = eval_dataset.shuffle(seed=42).select(range(eval_sample_size))

    print("数据集准备完成。")
    return train_dataset, eval_dataset, num_labels
