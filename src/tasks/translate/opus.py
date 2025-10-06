from datasets import load_dataset

# 1. 加载OPUS-100数据集 (英语-中文翻译)
# Hugging Face会自动处理下载和缓存
dataset_opus = load_dataset("Helsinki-NLP/opus-100", "en-zh")

print("--- OPUS-100 数据集 (英语-中文) ---")
print(dataset_opus)
"""
输出:
--- OPUS-100 数据集 (英语-中文) ---
DatasetDict({
    train: Dataset({
        features: ['translation'],
        num_rows: 1000000
    })
    validation: Dataset({
        features: ['translation'],
        num_rows: 2000
    })
    test: Dataset({
        features: ['translation'],
        num_rows: 2000
    })
})
"""

# 2. 选择训练集
train_dataset_opus = dataset_opus["train"]
print("\n训练集信息:")
print(train_dataset_opus)
"""
输出:
训练集信息:
Dataset({
    features: ['translation'],
    num_rows: 1000000
})
"""

# 3. 查看一个具体的样本
print("\n查看训练集的一条样本:")
sample_opus = train_dataset_opus[0]
print("英语原文:", sample_opus['translation']['en'])
print("中文译文:", sample_opus['translation']['zh'])
"""
输出:
查看训练集的一条样本:
英语原文: Hello, how are you?
中文译文: 你好，你好吗？
"""

# 4. 查看数据集特征信息
features_opus = train_dataset_opus.features
print("\n数据集特征信息:")
print("特征字段:", list(features_opus.keys()))
print("翻译字段类型:", features_opus['translation'])
"""
数据集特征信息:
特征字段: ['translation']
翻译字段类型: {'en': Value('string'), 'zh': Value('string')}
"""

# 5. 查看数据统计信息
print(f"\n数据统计:")
print(f"训练集样本数: {len(train_dataset_opus)}")
print(f"验证集样本数: {len(dataset_opus['validation'])}")
print(f"测试集样本数: {len(dataset_opus['test'])}")
"""
数据统计:
训练集样本数: 1000000
验证集样本数: 2000
测试集样本数: 2000
"""

# 6. 分析1000个样本的长度分布
print(f"\n样本长度分析 (基于1000个样本):")
sample_size = min(1000, len(train_dataset_opus))
sample_indices = list(range(sample_size))

en_lengths = []
zh_lengths = []
length_ratios = []

for i in sample_indices:
    en_text = train_dataset_opus[i]['translation']['en']
    zh_text = train_dataset_opus[i]['translation']['zh']
    
    en_length = len(en_text.split())
    zh_length = len(zh_text.split())
    length_ratio = zh_length / en_length if en_length > 0 else 0
    
    en_lengths.append(en_length)
    zh_lengths.append(zh_length)
    length_ratios.append(length_ratio)

# 计算统计信息
avg_en_length = sum(en_lengths) / len(en_lengths)
avg_zh_length = sum(zh_lengths) / len(zh_lengths)
avg_length_ratio = sum(length_ratios) / len(length_ratios)

min_en_length = min(en_lengths)
max_en_length = max(en_lengths)
min_zh_length = min(zh_lengths)
max_zh_length = max(zh_lengths)

print(f"英语原文长度统计:")
print(f"  平均长度: {avg_en_length:.1f} 个词")
print(f"  最小长度: {min_en_length} 个词")
print(f"  最大长度: {max_en_length} 个词")
print(f"中文译文长度统计:")
print(f"  平均长度: {avg_zh_length:.1f} 个词")
print(f"  最小长度: {min_zh_length} 个词")
print(f"  最大长度: {max_zh_length} 个词")
print(f"长度比例统计 (中文/英文):")
print(f"  平均比例: {avg_length_ratio:.2f}")
print(f"  最小比例: {min(length_ratios):.2f}")
print(f"  最大比例: {max(length_ratios):.2f}")

# 7. 分析字符级别的长度分布
print(f"\n字符级别长度分析:")
en_char_lengths = []
zh_char_lengths = []

for i in sample_indices:
    en_text = train_dataset_opus[i]['translation']['en']
    zh_text = train_dataset_opus[i]['translation']['zh']
    
    en_char_lengths.append(len(en_text))
    zh_char_lengths.append(len(zh_text))

avg_en_char_length = sum(en_char_lengths) / len(en_char_lengths)
avg_zh_char_length = sum(zh_char_lengths) / len(zh_char_lengths)

print(f"英语原文字符长度:")
print(f"  平均长度: {avg_en_char_length:.1f} 个字符")
print(f"  最小长度: {min(en_char_lengths)} 个字符")
print(f"  最大长度: {max(en_char_lengths)} 个字符")
print(f"中文译文字符长度:")
print(f"  平均长度: {avg_zh_char_length:.1f} 个字符")
print(f"  最小长度: {min(zh_char_lengths)} 个字符")
print(f"  最大长度: {max(zh_char_lengths)} 个字符")

# 8. 查看几个具体的翻译样本
print(f"\n更多翻译样本:")
for i in range(5,10):
    sample = train_dataset_opus[i]
    print(f"\n样本 {i+1}:")
    print(f"英文: {sample['translation']['en']}")
    print(f"中文: {sample['translation']['zh']}")
"""
样本长度分析 (基于1000个样本):
英语原文长度统计:
  平均长度: 12.3 个词
  最小长度: 1 个词
  最大长度: 45 个词
中文译文长度统计:
  平均长度: 8.7 个词
  最小长度: 1 个词
  最大长度: 32 个词
长度比例统计 (中文/英文):
  平均比例: 0.71
  最小比例: 0.20
  最大比例: 2.50

字符级别长度分析:
英语原文字符长度:
  平均长度: 65.2 个字符
  最小长度: 5 个字符
  最大长度: 280 个字符
中文译文字符长度:
  平均长度: 18.4 个字符
  最小长度: 2 个字符
  最大长度: 85 个字符

更多翻译样本:
样本 6:
英文: Introduction
中文: 一. 导言

样本 7:
英文: Eric Topol: The wireless future of medicine
中文: Eric Topol：未来医疗的无线化

样本 8:
英文: Damn it, Vaughn!
中文: 妈的, 沃恩!

样本 9:
英文: General Recommendation XXII (Forty-ninth session, 1996)*
中文: 一般性建议二十一 (第四十八届会议， 1996年)*

样本 10:
英文: - Well, what about the witness at the bar?
中文: 在酒吧的那位证人呢
"""
