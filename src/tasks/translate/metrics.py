import numpy as np
import evaluate  # 使用新版的 evaluate 库，它取代了 datasets.load_metric

# 加载BLEU评估指标
metric = evaluate.load("sacrebleu")


def compute_metrics(eval_preds, tokenizer):
    """
    计算并返回翻译任务的评估指标（主要是BLEU）。

    Args:
        eval_preds (tuple): Trainer返回的评估预测结果，包含 predictions 和 label_ids。
        tokenizer: Hugging Face Tokenizer实例。

    Returns:
        dict: 包含计算出的指标的字典。
    """
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    vocab_size = tokenizer.vocab_size
    # 检查是否有真正的错误（排除 -100，这是 transformers 内部使用的特殊值）
    invalid_mask = (preds < 0) | (preds >= vocab_size)
    # 排除 -100，因为这是 transformers 内部使用的特殊值
    invalid_mask = invalid_mask & (preds != -100)


    if np.any(invalid_mask):
        invalid_count = np.sum(invalid_mask)
        invalid_values = preds[invalid_mask]
        print(
            f"错误: 发现 {invalid_count} 个无效的 token ID，范围: [{np.min(invalid_values)}, {np.max(invalid_values)}]")
        print(f"词汇表大小: {vocab_size}")
        print("停止计算，返回空结果")
        return {}

    # 将模型生成的 token ID 解码为文本
    # skip_special_tokens=True 会移除像 <pad>, </s> 这样的特殊token
    preds = np.where(preds == -100, tokenizer.pad_token_id, preds)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # 将真实的标签 token ID 解码为文本
    # 对于标签中的 -100（被忽略的padding token），我们需要先替换为tokenizer的pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # sacrebleu 要求预测是字符串列表，而标签是字符串列表的列表
    # e.g., preds=["hello"], references=[["你好"]]
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    # 计算BLEU分数
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)

    # 也可以计算一下生成的平均长度
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]

    # 将结果打包成一个字典
    metrics_result = {
        "bleu": result["score"],
        "gen_len": np.mean(prediction_lens)
    }

    return {k: round(v, 4) for k, v in metrics_result.items()}
