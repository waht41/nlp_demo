# tasks/cnndm_summarization/data_handler.py
from datasets import load_dataset


def load_and_prepare_dataset(dataset_name, tokenizer, train_sample_size=None, eval_sample_size=None,
                             max_source_length=1024, max_target_length=128, **kwargs):
    """加载并预处理 CNN/DailyMail 数据集"""
    # 加载数据集
    dataset = load_dataset(dataset_name, kwargs.get('dataset_config_name', None))

    # 如果指定了样本大小，则进行采样
    if train_sample_size:
        dataset["train"] = dataset["train"].select(range(train_sample_size))
    if eval_sample_size:
        dataset["validation"] = dataset["validation"].select(range(eval_sample_size))

    def preprocess_function(examples):
        # 对输入文章进行分词 (只做截断，不做填充)
        model_inputs = tokenizer(examples["article"], max_length=max_source_length, truncation=True, padding=True)

        # 对目标摘要进行分词 (作为标签)
        with tokenizer.as_target_tokenizer():
            # 这里也不做填充
            labels = tokenizer(examples["highlights"], max_length=max_target_length, truncation=True, padding=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # 应用预处理函数
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["article", "highlights", "id"])

    # 对于S2S任务，num_labels 不适用，返回 None
    return tokenized_datasets["train"], tokenized_datasets["validation"], None