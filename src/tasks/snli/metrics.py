import numpy as np
import evaluate

# 加载评估模块
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    """
    计算并返回评估指标的函数，与Trainer API兼容。

    Args:
        eval_pred (tuple): 模型的预测结果和真实标签。

    Returns:
        dict: 包含各项指标的字典。
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # 计算准确率
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)

    # 计算F1分数 (使用宏平均，对类别不均衡问题更公平)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")

    # 将所有指标合并到一个字典中
    metrics = {}
    metrics.update(accuracy)
    metrics.update(f1)

    return metrics
