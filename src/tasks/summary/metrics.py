import evaluate
import numpy as np
import pysbd

# 加载 ROUGE 指标
rouge_metric = evaluate.load("rouge")
seg = pysbd.Segmenter(language="en", clean=False)

def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred

    # 清理 predictions 中的无效 token ID
    # 确保所有 token ID 都在有效范围内
    vocab_size = tokenizer.vocab_size
    
    # 检查并报告无效的 token ID
    invalid_mask = (predictions < 0) | (predictions >= vocab_size)
    if np.any(invalid_mask):
        invalid_count = np.sum(invalid_mask)
        invalid_values = predictions[invalid_mask]
        print(f"警告: 发现 {invalid_count} 个无效的 token ID，范围: [{np.min(invalid_values)}, {np.max(invalid_values)}]")
        print(f"词汇表大小: {vocab_size}")
    
    predictions = np.where(
        (predictions >= 0) & (predictions < vocab_size), 
        predictions, 
        tokenizer.pad_token_id
    )
    
    # 将生成的 token ID 解码成文本
    # -100 是一个特殊值，用于在标签中标记被忽略的 token（如 padding），需要替换掉
    try:
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    except OverflowError:
        print(f"OverflowError: 解码 predictions 时出错")
        print(f"predictions 形状: {predictions.shape}")
        print(f"predictions 数据类型: {predictions.dtype}")
        print(f"predictions 范围: [{np.min(predictions)}, {np.max(predictions)}]")
        raise
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE 需要句子之间有换行符
    decoded_preds = ["\n".join(seg.segment(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(seg.segment(label.strip())) for label in decoded_labels]

    # 计算 ROUGE 分数
    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # 提取关键结果
    result = {key: value * 100 for key, value in result.items()}

    # 添加每个摘要的平均长度
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}