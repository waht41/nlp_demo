import math
from torch.nn import CrossEntropyLoss
import torch

def compute_metrics(eval_preds):
    """
    为 Causal LM 任务计算评估指标。
    主要指标是困惑度 (Perplexity)。

    Args:
        eval_preds: Trainer 在评估期间返回的 EvalPrediction 对象，
                    包含 predictions (logits) 和 label_ids。

    Returns:
        dict: 包含指标名称和值的字典。
    """
    logits, labels = eval_preds

    # Trainer for CausalLM 已经处理了 logits 和 labels 的移位
    # 我们只需要计算交叉熵损失即可
    # PyTorch 的 CrossEntropyLoss 会自动忽略 labels 中为 -100 的部分


    # 将 logits 和 labels 转换为 PyTorch 张量
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    # Reshape logits: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
    # Reshape labels: (batch_size, seq_len) -> (batch_size * seq_len)
    shift_logits = logits.view(-1, logits.size(-1))
    shift_labels = labels.view(-1)

    loss_fct = CrossEntropyLoss()
    loss = loss_fct(shift_logits, shift_labels)

    try:
        # PPL = exp(cross_entropy_loss)
        perplexity = math.exp(loss.item())
    except OverflowError:
        perplexity = float("inf")

    # main.py 的 summary 日志会记录 'eval_loss'，这里我们返回 PPL
    return {"perplexity": perplexity}