from datasets import load_dataset

# 1. 加载SNLI数据集
# Hugging Face会自动处理下载和缓存
dataset_snli = load_dataset("snli")

print("--- SNLI 数据集 ---")
print(dataset_snli)
"""
输出:
--- SNLI 数据集 ---
DatasetDict({
    test: Dataset({
        features: ['premise', 'hypothesis', 'label'],
        num_rows: 10000
    })
    validation: Dataset({
        features: ['premise', 'hypothesis', 'label'],
        num_rows: 10000
    })
    train: Dataset({
        features: ['premise', 'hypothesis', 'label'],
        num_rows: 550152
    })
})
"""

# 2. 选择训练集
train_dataset_snli = dataset_snli["train"]
print("\n训练集信息:")
print(train_dataset_snli)
"""
输出:
训练集信息:
Dataset({
    features: ['premise', 'hypothesis', 'label'],
    num_rows: 550152
})
"""

# 3. 查看一个具体的样本
print("\n查看训练集的一条样本:")
sample_snli = train_dataset_snli[0]
print(sample_snli)
"""
输出:
查看训练集的一条样本:
{
  'premise': 'A person on a horse jumps over a broken down airplane.', 
  'hypothesis': 'A person is training his horse for a competition.', 
  'label': 1
}
# 注意: 数据结构非常清晰，直接包含 premise, hypothesis 和 label
# 这里的 label=1 对应 "neutral"
"""

# 4. 查看label和名称的对应关系
features_snli = train_dataset_snli.features
print("\n标签信息:")
print(features_snli['label'])

# SNLI有一个特殊情况：label为-1表示无共识，在处理时通常会过滤掉
# 不过ClassLabel特性里只包含0, 1, 2
label_names_snli = features_snli['label'].names
print(f"标签类别总数: {len(label_names_snli)}")
print("标签ID与名称对应关系:")
print(f"  标签 0 对应: {label_names_snli[0]}")
print(f"  标签 1 对应: {label_names_snli[1]}")
print(f"  标签 2 对应: {label_names_snli[2]}")
"""
输出:
标签信息:
ClassLabel(names=['entailment', 'neutral', 'contradiction'], id=None)
标签类别总数: 3
标签ID与名称对应关系:
  标签 0 对应: entailment
  标签 1 对应: neutral
  标签 2 对应: contradiction
"""