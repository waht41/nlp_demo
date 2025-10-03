from datasets import load_dataset

# 1. 加载整个数据集（包含train, validation, test）
dataset = load_dataset("rotten_tomatoes")

# 打印数据集结构
print(dataset)
"""
输出:
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 8530
    })
    validation: Dataset({
        features: ['text', 'label'],
        num_rows: 1066
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 1066
    })
})
"""

# 2. 选择一个子集（例如训练集）
train_dataset = dataset["train"]
print("\n训练集信息:")
print(train_dataset)
"""
输出:
训练集信息:
Dataset({
    features: ['text', 'label'],
    num_rows: 8530
})
"""

# 3. 查看一个具体的样本
print("\n查看训练集的第一条样本:")
sample = train_dataset[0]
print(sample)
print('label type',type(sample['label']))
"""
输出:
查看训练集的第一条样本:
{
  'text': 'the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .', 
  'label': 1, 
}
"""

# 4. 查看label和名称的对应关系
features = train_dataset.features
print("\n标签信息:")
print(features['label'])
# 获取标签名称
label_names = features['label'].names
print(f"标签 0 对应: {label_names[0]}")
print(f"标签 1 对应: {label_names[1]}")
"""
输出:
标签信息:
ClassLabel(names=['neg', 'pos'], id=None)
标签 0 对应: neg
标签 1 对应: pos
"""