from datasets import load_dataset
import numpy as np

# 1. 加载 UltraChat 200k 数据集
# Hugging Face会自动处理下载和缓存
# 该数据集主要包含用于监督式微调（SFT）的训练集和测试集
dataset_chat = load_dataset("HuggingFaceH4/ultrachat_200k")

print("--- UltraChat 200k 数据集 ---")
print(dataset_chat)
"""
输出:
--- UltraChat 200k 数据集 ---
DatasetDict({
    train_sft: Dataset({
        features: ['prompt', 'prompt_id', 'messages'],
        num_rows: 207865
    })
    test_sft: Dataset({
        features: ['prompt', 'prompt_id', 'messages'],
        num_rows: 23110
    })
    train_gen: Dataset({
        features: ['prompt', 'prompt_id', 'messages'],
        num_rows: 256032
    })
    test_gen: Dataset({
        features: ['prompt', 'prompt_id', 'messages'],
        num_rows: 28304
    })
})
"""

# 2. 选择SFT训练集 (Supervised Fine-Tuning)
# 这与原文中的摘要任务最为相似
train_dataset_chat = dataset_chat["train_sft"]
print("\nSFT训练集信息:")
print(train_dataset_chat)
"""
输出:
SFT训练集信息:
Dataset({
    features: ['prompt', 'prompt_id', 'messages'],
    num_rows: 207865
})
"""

# 3. 查看一个具体的样本
print("\n查看训练集的一条样本:")
sample_chat = train_dataset_chat[0]
print("样本ID (prompt_id):", sample_chat['prompt_id'])
# 'prompt'字段是对话的初始指令
print("初始指令 (prompt):", sample_chat['prompt'][:200] + "...")
# 'messages'字段是一个包含完整对话的列表
print("对话内容 (前两轮):")
for msg in sample_chat['messages'][:2]:
    print(f"  - 角色: {msg['role']}, 内容: {msg['content'][:150]}...")
"""
输出:
查看训练集的一条样本:
样本ID (prompt_id): f0e37e9f7800261167ce91143f98f511f768847236f133f2d0aed60b444ebe57
初始指令 (prompt): These instructions apply to section-based themes (Responsive 6.0+, Retina 4.0+, Parallax 3.0+ Turbo 2.0+, Mobilia 5.0+). What theme version am I using?
On your Collections pages & Featured Collections...
对话内容 (前两轮):
  - 角色: user, 内容: These instructions apply to section-based themes (Responsive 6.0+, Retina 4.0+, Parallax 3.0+ Turbo 2.0+, Mobilia 5.0+). What theme version am I using...
  - 角色: assistant, 内容: This feature only applies to Collection pages and Featured Collections sections of the section-based themes listed in the text material....
"""

# 4. 查看数据集特征信息
features_chat = train_dataset_chat.features
print("\n数据集特征信息:")
print("特征字段:", list(features_chat.keys()))
print("初始指令 'prompt' 字段类型:", features_chat['prompt'])
print("对话 'messages' 字段类型:", features_chat['messages'])
print("ID 'prompt_id' 字段类型:", features_chat['prompt_id'])
"""
输出:
数据集特征信息:
特征字段: ['prompt', 'prompt_id', 'messages']
初始指令 'prompt' 字段类型: Value('string')
对话 'messages' 字段类型: List({'content': Value('string'), 'role': Value('string')})
ID 'prompt_id' 字段类型: Value('string')
"""

# 5. 查看数据统计信息
print(f"\n数据统计:")
print(f"SFT 训练集样本数: {len(train_dataset_chat)}")
print(f"SFT 测试集样本数: {len(dataset_chat['test_sft'])}")
"""
输出:
数据统计:
SFT 训练集样本数: 207865
SFT 测试集样本数: 23110
"""

# 6. 分析1000个样本的长度分布
# 对于对话数据集，我们分析用户提问的总长度和助手回答的总长度
print(f"\n样本长度分析 (基于1000个样本):")
sample_size = min(1000, len(train_dataset_chat))
sample_indices = list(range(sample_size))

user_lengths = []
assistant_lengths = []
response_ratios = []  # 提问长度 / 回答长度

for i in sample_indices:
    messages = train_dataset_chat[i]['messages']
    user_len = 0
    assistant_len = 0
    for msg in messages:
        if msg['role'] == 'user':
            user_len += len(msg['content'].split())
        elif msg['role'] == 'assistant':
            assistant_len += len(msg['content'].split())

    # 避免除以零
    ratio = user_len / assistant_len if assistant_len > 0 else 0

    user_lengths.append(user_len)
    assistant_lengths.append(assistant_len)
    response_ratios.append(ratio)

# 计算统计信息
avg_user_length = np.mean(user_lengths)
avg_assistant_length = np.mean(assistant_lengths)
avg_response_ratio = np.mean(response_ratios)

min_user_length = np.min(user_lengths)
max_user_length = np.max(user_lengths)
min_assistant_length = np.min(assistant_lengths)
max_assistant_length = np.max(assistant_lengths)

print(f"用户消息总长度统计:")
print(f"  平均长度: {avg_user_length:.1f} 个词")
print(f"  最小长度: {min_user_length} 个词")
print(f"  最大长度: {max_user_length} 个词")
print(f"助手回应总长度统计:")
print(f"  平均长度: {avg_assistant_length:.1f} 个词")
print(f"  最小长度: {min_assistant_length} 个词")
print(f"  最大长度: {max_assistant_length} 个词")
print(f"回应比例统计 (用户/助手):")
print(f"  平均比例: {avg_response_ratio:.2f}:1")
print(f"  最小比例: {min(response_ratios):.2f}:1")
print(f"  最大比例: {max(response_ratios):.2f}:1")
"""
输出:
样本长度分析 (基于1000个样本):
用户消息总长度统计:
  平均长度: 199.9 个词
  最小长度: 22 个词
  最大长度: 1683 个词
助手回应总长度统计:
  平均长度: 704.7 个词
  最小长度: 39 个词
  最大长度: 2729 个词
回应比例统计 (用户/助手):
  平均比例: 0.56:1
  最小比例: 0.03:1
  最大比例: 7.49:1
"""