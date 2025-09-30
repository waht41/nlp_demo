# model_handler.py
# 负责加载预训练模型和分词器

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_model_and_tokenizer(model_checkpoint, num_labels):
    """
    加载指定checkpoint的模型和分词器。

    Args:
        model_checkpoint (str): 预训练模型的名称 (例如 "distilbert-base-uncased")。
        num_labels (int): 数据集的类别数量。

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"正在从 '{model_checkpoint}' 加载模型和分词器...")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # 加载模型，并指定分类任务的标签数量
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels
    )

    print("模型和分词器加载完成。")
    return model, tokenizer
