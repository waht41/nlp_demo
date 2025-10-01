import os
import yaml
import shutil
import subprocess
import argparse
import pandas as pd
from datetime import datetime

from transformers import TrainingArguments, Trainer, AutoTokenizer

from model_handler import load_model_and_tokenizer
from data_handler import load_and_prepare_dataset
from metrics import compute_metrics


def get_git_info(output_dir):
    """获取当前的 git commit hash 和未提交的修改，并保存到结果目录"""
    try:
        # 1. 获取 git commit hash
        git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')

        # 2. 将未提交的修改保存为一个 patch 文件
        # 这对于复现至关重要，因为它记录了在最后一次 commit 之后的所有代码改动
        diff_path = os.path.join(output_dir, "code_changes.patch")
        with open(diff_path, "w") as f:
            subprocess.run(['git', 'diff', 'HEAD'], stdout=f)

        print(f"✅ Git Hash ({git_hash}) 已记录。")
        if os.path.getsize(diff_path) > 0:
            print(f"⚠️ 发现未提交的代码修改，已保存至: {diff_path}")
        else:
            os.remove(diff_path)  # 如果没有改动，则删除空的 patch 文件

        return git_hash
    except subprocess.CalledProcessError:
        print("❓ 未能获取 Git 信息。可能不是一个 Git 仓库。")
        return "N/A"


def main(config_path: str):
    """主函数，从配置文件执行完整的训练和评估流程"""

    # --- 1. 加载配置 ---
    print(f"📖 从 '{config_path}' 加载配置...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 从配置中提取参数
    metadata_cfg = config.get('metadata', {})
    model_data_cfg = config.get('model_data', {})
    training_cfg = config.get('training', {})

    # --- 2. 创建唯一的实验目录和ID ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{timestamp}_{metadata_cfg.get('run_name', 'unnamed_run')}"

    # 结果和日志将保存在以 run_id 命名的专属文件夹中
    OUTPUT_DIR = os.path.join("./results", run_id)
    LOGGING_DIR = os.path.join("./logs", run_id)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)
    print(f"🚀 创建实验运行 ID: {run_id}")
    print(f"   - 模型和评估结果将保存至: {OUTPUT_DIR}")
    print(f"   - 配置文件、代码差异和TensorBoard日志将保存至: {LOGGING_DIR}")

    # --- 3. 记录代码和配置状态 (为了100%可复现) ---
    # 3.1 复制配置文件到日志目录
    shutil.copy(config_path, os.path.join(LOGGING_DIR, "config.yaml"))

    # 3.2 获取 Git 状态并保存代码差异
    git_hash = get_git_info(LOGGING_DIR)

    # --- 4. 加载数据集 ---
    print("\n" + "=" * 20 + " 正在加载数据集 " + "=" * 20)
    tokenizer = AutoTokenizer.from_pretrained(model_data_cfg['model_checkpoint'])

    train_dataset, eval_dataset, num_labels = load_and_prepare_dataset(
        dataset_name=model_data_cfg['dataset_name'],
        tokenizer=tokenizer,
        train_sample_size=model_data_cfg.get('train_sample_size'),
        eval_sample_size=model_data_cfg.get('eval_sample_size')
    )
    print(f"从数据集中推断出的 num_labels: {num_labels}")

    # --- 5. 加载模型 ---
    print("\n" + "=" * 20 + " 正在加载模型 " + "=" * 20)
    model, _ = load_model_and_tokenizer(
        model_checkpoint=model_data_cfg['model_checkpoint'],
        num_labels=num_labels
    )

    # --- 6. 配置训练参数 ---
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
        warmup_ratio=training_cfg.get('warmup_ratio', 0.0),

        # 评估、保存和日志策略
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_strategy="steps",
        logging_steps=50,
    )

    # --- 7. 初始化并启动训练器 ---
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

    trainer.train()

    print("\n" + "=" * 40)
    print("          ✅ 训练完成 ✅          ")
    print("=" * 40 + "\n")

    # --- 8. 在评估集上进行最终评估 ---
    print("在最终评估集上进行评估...")
    final_metrics = trainer.evaluate(eval_dataset)
    print("最终评估结果:")
    print(final_metrics)

    # --- 9. 将实验摘要记录到中央CSV文件 ---
    summary = {
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
        summary_df.to_csv(log_file, index=False)
    else:
        summary_df.to_csv(log_file, mode='a', header=False, index=False)

    print("\n" + "=" * 40)
    print(f"   📊 实验总结已记录到中央日志: {log_file}   ")
    print("=" * 40 + "\n")

    # --- 10. 保存最终模型 ---
    final_model_path = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(final_model_path)
    print(f"最佳模型已保存至: {final_model_path}")
    print(f"要查看训练日志，请在终端运行: tensorboard --logdir ./logs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 YAML 配置文件运行模型训练。")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="指向实验配置文件的路径 (例如: configs/baseline_experiment.yaml)"
    )
    args = parser.parse_args()
    main(args.config)
