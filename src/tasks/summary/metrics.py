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
    
    # 检查是否有真正的错误（排除 -100，这是 transformers 内部使用的特殊值）
    invalid_mask = (predictions < 0) | (predictions >= vocab_size)
    # 排除 -100，因为这是 transformers 内部使用的特殊值
    invalid_mask = invalid_mask & (predictions != -100)
    
    if np.any(invalid_mask):
        invalid_count = np.sum(invalid_mask)
        invalid_values = predictions[invalid_mask]
        print(f"错误: 发现 {invalid_count} 个无效的 token ID，范围: [{np.min(invalid_values)}, {np.max(invalid_values)}]")
        print(f"词汇表大小: {vocab_size}")
        print("停止计算，返回空结果")
        return {}
    
    # -100 是一个特殊值，用于在标签中标记被忽略的 token（如 padding），需要替换掉
    predictions = np.where(predictions == -100, tokenizer.pad_token_id, predictions)
    
    # 将生成的 token ID 解码成文本
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
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