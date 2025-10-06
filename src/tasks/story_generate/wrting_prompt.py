from datasets import load_dataset

# 1. 加载Writing Prompts数据集
# Hugging Face会自动处理下载和缓存
dataset_wp = load_dataset("euclaise/writingprompts")

print("--- Writing Prompts 数据集 ---")
print(dataset_wp)
"""
输出:
--- Writing Prompts 数据集 ---
DatasetDict({
    train: Dataset({
        features: ['prompt', 'story'],
        num_rows: 272600
    })
    test: Dataset({
        features: ['prompt', 'story'],
        num_rows: 15138
    })
    validation: Dataset({
        features: ['prompt', 'story'],
        num_rows: 15620
    })
})
"""

# 2. 选择训练集
train_dataset_wp = dataset_wp["train"]
print("\n训练集信息:")
print(train_dataset_wp)
"""
输出:
训练集信息:
Dataset({
    features: ['prompt', 'story'],
    num_rows: 272600
})
"""

# 3. 查看一个具体的样本
print("\n查看训练集的一条样本:")
sample_wp = train_dataset_wp[0]
print("样本索引: 0")
print("提示内容:", sample_wp['prompt'])
print("故事内容:", sample_wp['story'][:200] + "...")
"""
输出:
查看训练集的一条样本:
样本索引: 0
提示内容: [ WP ] You 've finally managed to discover the secret to immortality . Suddenly , Death appears before you , hands you a business card , and says , `` When you realize living forever sucks , call this number , I 've got a job offer for you . ''

故事内容: So many times have I walked on ruins, the remainings of places that I loved and got used to.. At first I was scared, each time I could feel my city, my current generation collapse, break into the blac...
"""

# 4. 查看数据集特征信息
features_wp = train_dataset_wp.features
print("\n数据集特征信息:")
print("特征字段:", list(features_wp.keys()))
print("提示字段类型:", features_wp['prompt'])
print("故事字段类型:", features_wp['story'])
"""
数据集特征信息:
特征字段: ['prompt', 'story']
提示字段类型: Value('string')
故事字段类型: Value('string')
"""

# 5. 查看数据统计信息
print(f"\n数据统计:")
print(f"训练集样本数: {len(train_dataset_wp)}")
print(f"验证集样本数: {len(dataset_wp['validation'])}")
print(f"测试集样本数: {len(dataset_wp['test'])}")
"""
数据统计:
训练集样本数: 272600
验证集样本数: 15620
测试集样本数: 15138
"""

# 6. 分析1000个样本的长度分布
print(f"\n样本长度分析 (基于1000个样本):")
sample_size = min(1000, len(train_dataset_wp))
sample_indices = list(range(sample_size))

prompt_lengths = []
story_lengths = []
expansion_ratios = []

for i in sample_indices:
    prompt_length = len(train_dataset_wp[i]['prompt'].split())
    story_length = len(train_dataset_wp[i]['story'].split())
    expansion_ratio = story_length / prompt_length if prompt_length > 0 else 0
    
    prompt_lengths.append(prompt_length)
    story_lengths.append(story_length)
    expansion_ratios.append(expansion_ratio)

# 计算统计信息
avg_prompt_length = sum(prompt_lengths) / len(prompt_lengths)
avg_story_length = sum(story_lengths) / len(story_lengths)
avg_expansion_ratio = sum(expansion_ratios) / len(expansion_ratios)

min_prompt_length = min(prompt_lengths)
max_prompt_length = max(prompt_lengths)
min_story_length = min(story_lengths)
max_story_length = max(story_lengths)

print(f"提示长度统计:")
print(f"  平均长度: {avg_prompt_length:.1f} 个词")
print(f"  最小长度: {min_prompt_length} 个词")
print(f"  最大长度: {max_prompt_length} 个词")
print(f"故事长度统计:")
print(f"  平均长度: {avg_story_length:.1f} 个词")
print(f"  最小长度: {min_story_length} 个词")
print(f"  最大长度: {max_story_length} 个词")
print(f"扩展比统计:")
print(f"  平均扩展比: {avg_expansion_ratio:.2f}:1")
print(f"  最小扩展比: {min(expansion_ratios):.2f}:1")
print(f"  最大扩展比: {max(expansion_ratios):.2f}:1")
"""
样本长度分析 (基于1000个样本):
提示长度统计:
  平均长度: 29.0 个词
  最小长度: 4 个词
  最大长度: 74 个词
故事长度统计:
  平均长度: 553.3 个词
  最小长度: 100 个词
  最大长度: 1965 个词
扩展比统计:
  平均扩展比: 25.07:1
  最小扩展比: 1.88:1
  最大扩展比: 343.60:1
"""

# 7. 分析提示词类型分布
print(f"\n提示词类型分析 (基于1000个样本):")
prompt_types = {}
for i in sample_indices:
    prompt = train_dataset_wp[i]['prompt']
    # 提取提示词类型（通常是[WP]开头的格式）
    if prompt.startswith('[ WP ]'):
        prompt_type = 'Writing Prompt'
    elif prompt.startswith('[ EU ]'):
        prompt_type = 'EU'
    elif prompt.startswith('[ TT ]'):
        prompt_type = 'TT'
    elif prompt.startswith('[ CW ]'):
        prompt_type = 'CW'
    else:
        prompt_type = 'Other'
    
    prompt_types[prompt_type] = prompt_types.get(prompt_type, 0) + 1

print("提示词类型分布:")
for prompt_type, count in prompt_types.items():
    percentage = (count / sample_size) * 100
    print(f"  {prompt_type}: {count} 个 ({percentage:.1f}%)")
"""
提示词类型分析 (基于1000个样本):
提示词类型分布:
  Writing Prompt: 862 个 (86.2%)
  Other: 90 个 (9.0%)
  TT: 9 个 (0.9%)
  EU: 17 个 (1.7%)
  CW: 22 个 (2.2%)
"""
# 8. 分析故事质量指标
print(f"\n故事质量分析 (基于1000个样本):")

quality_metrics = {
    'very_short': 0,    # < 50词
    'short': 0,         # 50-200词
    'medium': 0,        # 200-500词
    'long': 0,          # 500-1000词
    'very_long': 0      # > 1000词
}

for story_length in story_lengths:
    if story_length < 50:
        quality_metrics['very_short'] += 1
    elif story_length < 200:
        quality_metrics['short'] += 1
    elif story_length < 500:
        quality_metrics['medium'] += 1
    elif story_length < 1000:
        quality_metrics['long'] += 1
    else:
        quality_metrics['very_long'] += 1

print("故事长度分布:")
for category, count in quality_metrics.items():
    percentage = (count / sample_size) * 100
    print(f"  {category}: {count} 个 ({percentage:.1f}%)")
"""
故事质量分析 (基于1000个样本):
故事长度分布:
  very_short: 0 个 (0.0%)
  short: 146 个 (14.6%)
  medium: 399 个 (39.9%)
  long: 334 个 (33.4%)
  very_long: 121 个 (12.1%)
"""
# 9. 查看几个完整的样本示例
print(f"\n完整样本示例:")
for i in range(min(3, len(train_dataset_wp))):
    sample = train_dataset_wp[i]
    print(f"\n--- 样本 {i+1} ---")
    print(f"索引: {i}")
    print(f"提示: {sample['prompt']}")
    print(f"故事: {sample['story'][:300]}...")
    print(f"故事总长度: {len(sample['story'].split())} 个词")
"""
--- 样本 1 ---
索引: 0
提示: [ WP ] You 've finally managed to discover the secret to immortality . Suddenly , Death appears before you , hands you a business card , and says , `` When you realize living forever sucks , call this number , I 've got a job offer for you . ''

故事: So many times have I walked on ruins, the remainings of places that I loved and got used to.. At first I was scared, each time I could feel my city, my current generation collapse, break into the black hole that thrives within it, I could feel humanity, the way I'm able to feel my body.. After a few...
故事总长度: 570 个词

--- 样本 2 ---
索引: 1
提示: [ WP ] The moon is actually a giant egg , and it has just started to hatch .

故事: -Week 18 aboard the Depth Reaver, Circa 2023- 
 
 I walk about the dull gray halls, the artificial gravity making my steps feel almost as if they were on land. Almost. I glance out a window as I pass it by. There's the sun, and there's the moon right there. And, of course, there's the Earth. I kinda...
故事总长度: 488 个词

--- 样本 3 ---
索引: 2
提示: [ WP ] You find a rip in time walking through the alleys . You enter it to find yourself on a metal table with surgical instruments on a chair next to you .

故事: I was feckin' sloshed, mate. First time I ever was in the Big Lemon, and I'd found me the best feckin' pub I could imagine, I tell ya what. So I stumble out when it was closin' time, musta been'round 4 o'clock in the morning, and made my way through some alleys to find the quaint little AirBnB place...
故事总长度: 467 个词
"""