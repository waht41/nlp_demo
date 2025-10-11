from datasets import load_dataset
import os


def load_and_prepare_dataset(dataset_name, tokenizer, train_sample_size=None, eval_sample_size=None, 
                           max_length=2048, **kwargs):
    """
    为 Mistral 模型加载并准备 UltraChat 200k 数据集用于预训练。
    
    Args:
        dataset_name (str): 数据集名称 (HuggingFaceH4/ultrachat_200k)
        tokenizer: Hugging Face Tokenizer 实例
        train_sample_size (int): 训练集采样大小
        eval_sample_size (int): 评估集采样大小
        max_length (int): 分词时的最大长度
        **kwargs: 其他参数
    
    Returns:
        tuple: (tokenized_train_dataset, tokenized_eval_dataset, None)
    """
    print(f"🔄 正在加载 UltraChat 200k 数据集...")
    
    # 加载 UltraChat 200k 数据集
    # 该数据集包含 train_sft, test_sft, train_gen, test_gen 四个子集
    dataset = load_dataset(dataset_name)
    
    # 使用 SFT (Supervised Fine-Tuning) 数据集
    # train_sft: 207,865 个样本用于训练
    # test_sft: 23,110 个样本用于评估
    train_dataset = dataset['train_sft']
    eval_dataset = dataset['test_sft']

    if train_sample_size:
        print(f"🔍 采样 {train_sample_size} 条数据作为训练集...")
        train_dataset = train_dataset.select(range(train_sample_size))
    if eval_sample_size:
        print(f"🔍 采样 {eval_sample_size} 条数据作为评估集...")
        eval_dataset = eval_dataset.select(range(eval_sample_size))

    print(f"训练集大小: {len(train_dataset)}, 评估集大小: {len(eval_dataset)}")

    if tokenizer.chat_template is None:
        print("🔧 Tokenizer 缺少聊天模板，正在手动设置 Mistral/Llama-2 格式的模板...")

        # 这是一个标准的 Jinja2 模板，用于格式化对话
        MISTRAL_CHAT_TEMPLATE = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<s>[INST] ' + message['content'] + ' [/INST]' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + '</s>' }}"
            "{% endif %}"
            "{% endfor %}"
        )
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
        print("✅ 聊天模板设置成功！")

    def tokenize_function(examples):
        """
        将对话消息转换为模型输入格式并分词。
        
        目标: 将对话消息转换为完整的输入序列用于预训练
        方法: 1. 使用 apply_chat_template 格式化对话
              2. 对格式化后的文本进行分词
              3. 返回 input_ids 用于预训练
        """
        # 1. 使用对话模板格式化消息
        formatted_texts = []
        for messages in examples['messages']:
            # apply_chat_template 将消息转换为模型需要的格式
            # 例如: "<s>[INST] user message [/INST] assistant response </s>"
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            formatted_texts.append(formatted_text)

        # 2. 对格式化后的文本进行分词
        model_inputs = tokenizer(
            formatted_texts,
            truncation=True,
            max_length=max_length,
            padding=False,  # 动态填充由 DataCollator 处理
        )
        
        return model_inputs

    print("🧠 ----------------------------------------------------")
    print("🧠 正在以对话格式对数据集进行分词...")
    print("🧠 (输入 = 格式化的对话, 标签由 DataCollator 自动生成)")
    print("🧠 ----------------------------------------------------")

    # 设置多线程处理，最多使用 8 个 CPU 核心
    num_proc = min(8, os.cpu_count())
    print(f"🔧 使用 {num_proc} 个进程进行并行处理...")

    tokenized_train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=num_proc,
        desc="处理训练集..."
    )
    
    tokenized_eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        num_proc=num_proc,
        desc="处理评估集..."
    )

    print("✅ 数据集准备完成！")
    return tokenized_train_dataset, tokenized_eval_dataset, None