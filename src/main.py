import os
import yaml
import shutil
import argparse
import pandas as pd
from datetime import datetime
import importlib  # 关键：用于动态导入模块

from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
from trainer_callback import DistributionLoggingCallback
from utils.git import get_git_info

# --- 主函数开始 ---
def main(task_name: str):
    """主函数，从指定的任务目录执行完整的训练和评估流程"""

    # --- 1. 动态加载任务模块和配置 ---
    print(f"🚀 开始执行任务: {task_name}")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TASKS_DIR = os.path.join(SCRIPT_DIR, "tasks")
    config_path = os.path.join(TASKS_DIR,task_name, "config.yaml")


    try:
        # 动态导入特定任务的数据处理和评估模块
        data_handler_module = importlib.import_module(f"tasks.{task_name}.data_handler")
        metrics_module = importlib.import_module(f"tasks.{task_name}.metrics")
    except ModuleNotFoundError:
        print(f"错误: 任务 '{task_name}' 不存在或其目录结构不完整。")
        print("请确保 tasks/ 目录下有对应的任务文件夹，且包含 data_handler.py 和 metrics.py。")
        return

    print(f"📖 从 '{config_path}' 加载配置...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    metadata_cfg = config.get('metadata', {})
    model_data_cfg = config.get('model_data', {})
    training_cfg = config.get('training', {})

    # --- 2. 创建唯一的实验目录和ID ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 让run_id包含任务名，更清晰
    run_id = f"{timestamp}_{task_name}_{metadata_cfg.get('run_name', 'unnamed_run')}"

    OUTPUT_DIR = os.path.join("./results", run_id)
    LOGGING_DIR = os.path.join("./logs", run_id)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)
    print(f"🚀 创建实验运行 ID: {run_id}")
    print(f"   - 结果将保存至: {OUTPUT_DIR}")
    print(f"   - 日志将保存至: {LOGGING_DIR}")

    # --- 3. 记录代码和配置状态 ---
    shutil.copy(config_path, os.path.join(LOGGING_DIR, "config.yaml"))
    git_hash = get_git_info(LOGGING_DIR)

    # --- 4. 加载数据集 (调用动态导入的模块) ---
    print("\n" + "=" * 20 + " 正在加载数据集 " + "=" * 20)
    tokenizer = AutoTokenizer.from_pretrained(model_data_cfg['model_checkpoint'])

    # 调用特定任务的 data_handler
    train_dataset, eval_dataset, num_labels = data_handler_module.load_and_prepare_dataset(
        dataset_name=model_data_cfg['dataset_name'],
        tokenizer=tokenizer,
        train_sample_size=model_data_cfg.get('train_sample_size'),
        eval_sample_size=model_data_cfg.get('eval_sample_size')
    )
    print(f"从数据集中推断出的 num_labels: {num_labels}")

    # --- 5. 加载模型 (可以简化为一个通用函数) ---
    print("\n" + "=" * 20 + " 正在加载模型 " + "=" * 20)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_data_cfg['model_checkpoint'],
        num_labels=num_labels
    )

    # --- 6. 配置训练参数 (这部分完全通用，无需修改) ---
    print("\n" + "=" * 20 + " 正在配置训练参数 " + "=" * 20)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=LOGGING_DIR,
        report_to="tensorboard",
        run_name=run_id,

        # 从配置文件中读取训练参数
        num_train_epochs=training_cfg['num_train_epochs'],
        per_device_train_batch_size=training_cfg['per_device_train_batch_size'],
        per_device_eval_batch_size=training_cfg.get('per_device_eval_batch_size',
                                                    training_cfg['per_device_train_batch_size']),
        learning_rate=float(training_cfg['learning_rate']),
        weight_decay=training_cfg.get('weight_decay', 0.0),
        max_grad_norm=training_cfg.get('max_grad_norm', 1.0),
        warmup_ratio=training_cfg.get('warmup_ratio', 0.0),  # 从配置文件读取 warmup ratio
        lr_scheduler_type=training_cfg.get('lr_scheduler_type', 'linear'),  # 从配置文件读取学习率调度类型

        # 评估、保存和日志策略
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_strategy="steps",
        logging_steps=50,
    )

    # --- 7. 初始化并启动训练器 (调用动态导入的模块) ---
    callbacks = []
    if training_cfg.get('log_distribution', False):
        print("📊 启用参数和梯度分布记录...")
        callbacks.append(DistributionLoggingCallback())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=metrics_module.compute_metrics,  # 调用特定任务的 compute_metrics
        callbacks=callbacks
    )

    # ... [你原来的训练、评估、保存总结的逻辑完全不变] ...
    print("\n" + "=" * 40 + "\n          🔥 开始模型训练 🔥          \n" + "=" * 40 + "\n")
    trainer.train()
    print("\n" + "=" * 40 + "\n          ✅ 训练完成 ✅          \n" + "=" * 40 + "\n")

    print("在最终评估集上进行评估...")
    final_metrics = trainer.evaluate(eval_dataset)
    print("最终评估结果:", final_metrics)

    # ... [保存 summary 到 experiments.csv 的逻辑] ...
    summary = {
        'task': task_name,
        'run_id': run_id,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'run_name': metadata_cfg.get('run_name', ''),
        'description': metadata_cfg.get('description', ''),
        'git_hash': git_hash,
        'model_checkpoint': model_data_cfg['model_checkpoint'],
        'dataset_name': model_data_cfg['dataset_name'],
        'learning_rate': training_cfg['learning_rate'],
        'epochs': training_cfg['num_train_epochs'],
        'batch_size': training_cfg['per_device_train_batch_size'],
        'final_eval_accuracy': final_metrics.get('eval_accuracy'),
        'final_eval_loss': final_metrics.get('eval_loss'),
        'results_path': OUTPUT_DIR,
    }

    log_file = "./experiments.csv"
    summary_df = pd.DataFrame([summary])

    # 线程安全地追加到 CSV 文件
    if not os.path.exists(log_file):
        summary_df.to_csv(log_file, index=False, encoding='utf-8-sig')
    else:
        summary_df.to_csv(log_file, mode='a', header=False, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 40)
    print(f"   📊 实验总结已记录到中央日志: {log_file}   ")
    print("=" * 40 + "\n")

    # --- 10. 保存最终模型 ---
    final_model_path = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(final_model_path)
    print(f"最佳模型已保存至: {final_model_path}")
    print(f"要查看训练日志，请在终端运行: tensorboard --logdir ./logs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从指定的任务目录运行模型训练。")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="要执行的任务名称 (必须是 tasks/ 目录下的一个子文件夹名，例如: rotten_tomatoes)"
    )
    args = parser.parse_args()
    main(args.task)