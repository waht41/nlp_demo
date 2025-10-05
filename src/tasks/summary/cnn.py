from datasets import load_dataset

# 1. 加载CNN/DailyMail数据集
# Hugging Face会自动处理下载和缓存
dataset_cnn = load_dataset("cnn_dailymail", "3.0.0")

print("--- CNN/DailyMail 数据集 ---")
print(dataset_cnn)
"""
输出:
--- CNN/DailyMail 数据集 ---
DatasetDict({
    train: Dataset({
        features: ['article', 'highlights', 'id'],
        num_rows: 287113
    })
    validation: Dataset({
        features: ['article', 'highlights', 'id'],
        num_rows: 13368
    })
    test: Dataset({
        features: ['article', 'highlights', 'id'],
        num_rows: 11490
    })
})
"""

# 2. 选择训练集
train_dataset_cnn = dataset_cnn["train"]
print("\n训练集信息:")
print(train_dataset_cnn)
"""
输出:
训练集信息:
Dataset({
    features: ['article', 'highlights', 'id'],
    num_rows: 287113
})
"""

# 3. 查看一个具体的样本
print("\n查看训练集的一条样本:")
sample_cnn = train_dataset_cnn[0]
print("文章ID:", sample_cnn['id'])
print("文章内容:", sample_cnn['article'][:200] + "...")
print("摘要内容:", sample_cnn['highlights'])
"""
输出:
查看训练集的一条样本:
文章ID: 42c027e4ff9730fbb3de84c1af0d2c506e41c3e4
文章内容: LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on ...
摘要内容: Harry Potter star Daniel Radcliffe gets £20M fortune as he turns 18 Monday .
Young actor says he has no plans to fritter his cash away .
Radcliffe's earnings from first five Potter films have been held in trust fund .
"""

# 4. 查看数据集特征信息
features_cnn = train_dataset_cnn.features
print("\n数据集特征信息:")
print("特征字段:", list(features_cnn.keys()))
print("文章字段类型:", features_cnn['article'])
print("摘要字段类型:", features_cnn['highlights'])
print("ID字段类型:", features_cnn['id'])
"""
数据集特征信息:
特征字段: ['article', 'highlights', 'id']
文章字段类型: Value('string')
摘要字段类型: Value('string')
ID字段类型: Value('string')
"""
# 5. 查看数据统计信息
print(f"\n数据统计:")
print(f"训练集样本数: {len(train_dataset_cnn)}")
print(f"验证集样本数: {len(dataset_cnn['validation'])}")
print(f"测试集样本数: {len(dataset_cnn['test'])}")
"""
数据统计:
训练集样本数: 287113
验证集样本数: 13368
测试集样本数: 11490
"""

# 6. 分析1000个样本的长度分布
print(f"\n样本长度分析 (基于1000个样本):")
sample_size = min(1000, len(train_dataset_cnn))
sample_indices = list(range(sample_size))

article_lengths = []
highlights_lengths = []
compression_ratios = []

for i in sample_indices:
    article_length = len(train_dataset_cnn[i]['article'].split())
    highlights_length = len(train_dataset_cnn[i]['highlights'].split())
    compression_ratio = article_length / highlights_length if highlights_length > 0 else 0
    
    article_lengths.append(article_length)
    highlights_lengths.append(highlights_length)
    compression_ratios.append(compression_ratio)

# 计算统计信息
avg_article_length = sum(article_lengths) / len(article_lengths)
avg_highlights_length = sum(highlights_lengths) / len(highlights_lengths)
avg_compression_ratio = sum(compression_ratios) / len(compression_ratios)

min_article_length = min(article_lengths)
max_article_length = max(article_lengths)
min_highlights_length = min(highlights_lengths)
max_highlights_length = max(highlights_lengths)

print(f"文章长度统计:")
print(f"  平均长度: {avg_article_length:.1f} 个词")
print(f"  最小长度: {min_article_length} 个词")
print(f"  最大长度: {max_article_length} 个词")
print(f"摘要长度统计:")
print(f"  平均长度: {avg_highlights_length:.1f} 个词")
print(f"  最小长度: {min_highlights_length} 个词")
print(f"  最大长度: {max_highlights_length} 个词")
print(f"压缩比统计:")
print(f"  平均压缩比: {avg_compression_ratio:.2f}:1")
print(f"  最小压缩比: {min(compression_ratios):.2f}:1")
print(f"  最大压缩比: {max(compression_ratios):.2f}:1")
"""
样本长度分析 (基于1000个样本):
文章长度统计:
  平均长度: 591.6 个词
  最小长度: 50 个词
  最大长度: 1743 个词
摘要长度统计:
  平均长度: 42.8 个词
  最小长度: 23 个词
  最大长度: 66 个词
压缩比统计:
  平均压缩比: 14.15:1
  最小压缩比: 1.92:1
  最大压缩比: 61.12:1
"""