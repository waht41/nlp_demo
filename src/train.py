# train.py
# 主训练脚本，整合所有模块并启动训练
from datetime import datetime

from transformers import TrainingArguments, Trainer, AutoTokenizer

# 从我们创建的模块中导入函数
from src.model_handler import load_model_and_tokenizer
from src.data_handler import load_and_prepare_dataset
from src.metrics import compute_metrics


def main():
    """主函数，执行完整的训练和评估流程"""

    # --- 1. 定义配置 ---
    # 模型和数据集配置
    MODEL_CHECKPOINT = "distilbert-base-uncased"
    DATASET_NAME = "rotten_tomatoes"

    # 快速测试的采样大小 (设置为None可使用完整数据集)
    TRAIN_SAMPLE_SIZE = None
    EVAL_SAMPLE_SIZE = None

    # 为日志创建名称
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    learning_rate = 1e-6
    run_name = f"lr-{learning_rate}_{timestamp}"

    # 目录配置 - 为每个训练运行创建独立文件夹
    OUTPUT_DIR = f"./results/{run_name}"
    LOGGING_DIR = f"./logs/{run_name}"


    # --- 2. 加载数据集 ---
    # 先加载一次分词器，用于数据预处理
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    train_dataset, eval_dataset, num_labels = load_and_prepare_dataset(
        dataset_name=DATASET_NAME,
        tokenizer=tokenizer,
        train_sample_size=TRAIN_SAMPLE_SIZE,
        eval_sample_size=EVAL_SAMPLE_SIZE
    )
    print(f"从数据集中推断出的 num_labels: {num_labels}")

    print("\n" + "=" * 20 + " 最终数据集验证 " + "=" * 20)
    print("--- 训练集信息 ---")
    print(train_dataset)
    print("训练集特征 (Features):")
    print(train_dataset.features)  # 重点检查这里，确认有 'labels' 且类型正确

    print("\n--- 抽查第一个样本 ---")
    sample = train_dataset[0]
    print(sample)
    # 确认 'labels' 键存在，并且值是一个 tensor 整数
    print(f"样本中 'labels' 的值: {sample['labels']}")
    print(f"样本中 'labels' 的数据类型: {sample['labels'].dtype}")
    print("=" * 40 + "\n")
    # ===============================================================

    # --- 3. 加载模型 ---
    # `_` 接收了再次加载的分词器，因为我们已经有了，所以忽略它
    model, _ = load_model_and_tokenizer(
        model_checkpoint=MODEL_CHECKPOINT,
        num_labels=num_labels
    )

    # --- 4. 配置训练参数 ---
    print("配置训练参数...")
    training_args = TrainingArguments(
        # 目录和报告配置
        output_dir=OUTPUT_DIR,
        logging_dir=LOGGING_DIR,
        report_to="tensorboard",  # 启用TensorBoard日志
        run_name=run_name,

        # 训练过程配置
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=learning_rate,
        weight_decay=0.01,
        max_grad_norm=1.0,  # 梯度裁剪
        warmup_ratio=0.1,

        # 评估和保存策略
        eval_strategy="epoch",  # 每个epoch结束后进行评估
        save_strategy="epoch",  # 每个epoch结束后保存模型
        load_best_model_at_end=True,  # 训练结束后加载最佳模型

        # 日志记录
        logging_strategy="steps",
        logging_steps=10,  # 每10步记录一次日志到控制台和TensorBoard
    )

    # --- 5. 初始化并启动训练器 ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\n" + "=" * 40)
    print("          🔥 开始模型训练 🔥          ")
    print("=" * 40 + "\n")

    # 启动训练
    trainer.train()

    print("\n" + "=" * 40)
    print("          ✅ 训练完成 ✅          ")
    print("=" * 40 + "\n")

    # --- 6. 在评估集上进行最终评估 ---
    print("在最终评估集上进行评估...")
    final_metrics = trainer.evaluate(eval_dataset)
    print("最终评估结果:")
    print(final_metrics)

    # --- 7. 保存最终模型 ---
    final_model_path = f"{OUTPUT_DIR}/final_model"
    trainer.save_model(final_model_path)
    print(f"\n最佳模型已保存至: {final_model_path}")
    print(f"要查看训练日志，请在终端运行: tensorboard --logdir {LOGGING_DIR}")


if __name__ == "__main__":
    main()
