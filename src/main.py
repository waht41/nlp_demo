import os
import yaml
import shutil
import argparse
import pandas as pd
from datetime import datetime
import importlib
from functools import partial
from transformers import (
    TrainingArguments, Trainer, 
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq
)
from trainer_callback import DistributionLoggingCallback
from utils.git import get_git_info

# --- 主函数开始 ---
def main(task_name: str, resume_from: str = None):
    """主函数，从指定的任务目录执行完整的训练和评估流程
    
    Args:
        task_name: 要执行的任务名称
        resume_from: 可选，指定要从哪个检查点继续训练
    """

    # --- 1. 动态加载任务模块和配置 ---
    print(f"🚀 开始执行任务: {task_name}")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TASKS_DIR = os.path.join(SCRIPT_DIR, "tasks")
    config_path = os.path.join(TASKS_DIR,task_name, "config.yaml")


    try:
        # 动态导入特定任务的数据处理和评估模块
        data_handler_module = importlib.import_module(f"tasks.{task_name}.data_handler")
        metrics_module = importlib.import_module(f"tasks.{task_name}.metrics")
    except ModuleNotFoundError as e:
        print(f"错误: 导入任务 '{task_name}' 相关模块时失败。")
        print(f"具体错误: {str(e)}")
        print("可能的原因:")
        print("1. tasks/ 目录下没有对应的任务文件夹")
        print("2. 任务文件夹中缺少必需的 data_handler.py 或 metrics.py 文件")
        print("3. 任务模块中引用的依赖包未安装，请检查 requirements.txt 并安装所需依赖")
        return

    print(f"📖 从 '{config_path}' 加载配置...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    metadata_cfg = config.get('metadata', {})
    model_data_cfg = config.get('model_data', {})
    training_cfg = config.get('training', {})

    # 读取任务类型，这是关键！
    task_type = model_data_cfg.get('task_type', 'classification')
    if task_type and task_type not in ['classification', 'seq2seq']:
        print('未知任务类型')
        return

    print(f"检测到任务类型: {task_type}")

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

    # 这里 data_handler.py 内部会处理不同任务的逻辑
    # 封装通用参数，避免重复
    common_dataset_args = {
        'dataset_name': model_data_cfg['dataset_name'],
        'tokenizer': tokenizer,
        'train_sample_size': model_data_cfg.get('train_sample_size'),
        'eval_sample_size': model_data_cfg.get('eval_sample_size'),
        'dataset_config_name': model_data_cfg.get('dataset_config_name'),
    }
    
    if task_type == 'seq2seq':
        # 只有 seq2seq 任务才传递这两个参数
        datasets_and_labels = data_handler_module.load_and_prepare_dataset(
            **common_dataset_args,
            max_source_length=model_data_cfg.get('max_source_length'), # 为 S2S 任务增加参数
            max_target_length=model_data_cfg.get('max_target_length')  # 为 S2S 任务增加参数
        )
    else:
        # 分类任务使用原有的参数
        datasets_and_labels = data_handler_module.load_and_prepare_dataset(**common_dataset_args)
    # 对于分类任务，返回 (train, eval, num_labels)
    # 对于S2S任务，可以约定返回 (train, eval, None) 因为 num_labels 不适用
    train_dataset, eval_dataset, num_labels = datasets_and_labels
    if num_labels:
        print(f"从数据集中推断出的 num_labels: {num_labels}")

    # --- 5. 根据 task_type 加载模型 ---
    print("\n" + "=" * 20 + " 正在加载模型 " + "=" * 20)
    model_path = resume_from if resume_from else model_data_cfg['model_checkpoint']
    print(f"📂 从 '{model_path}' 加载...")

    if task_type == 'seq2seq':
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    else: # 默认为 classification
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

    # --- 6. 根据 task_type 配置训练参数 ---
    print("\n" + "=" * 20 + " 正在配置训练参数 " + "=" * 20)
    
    # 通用参数
    common_args = {
        'output_dir': OUTPUT_DIR,
        'logging_dir': LOGGING_DIR,
        'report_to': "tensorboard",
        'run_name': run_id,
        'num_train_epochs': training_cfg['num_train_epochs'],
        'per_device_train_batch_size': training_cfg['per_device_train_batch_size'],
        'per_device_eval_batch_size': training_cfg.get('per_device_eval_batch_size', training_cfg['per_device_train_batch_size']),
        'learning_rate': float(training_cfg['learning_rate']),
        'weight_decay': training_cfg.get('weight_decay', 0.0),
        'max_grad_norm': training_cfg.get('max_grad_norm', 1.0),
        'warmup_ratio': training_cfg.get('warmup_ratio', 0.0),
        'lr_scheduler_type': training_cfg.get('lr_scheduler_type', 'linear'),
        'eval_strategy': "epoch",
        'save_strategy': "epoch",
        'load_best_model_at_end': True,
        'logging_strategy': "steps",
        'logging_steps': training_cfg.get('logging_steps', 50),
        'fp16': training_cfg.get('fp16', True),
        'torch_compile': training_cfg.get('torch_compile', False),
    }

    if task_type == 'seq2seq':
        # S2S 任务特有的参数
        seq2seq_extra_args = {
            'predict_with_generate': True,
            'generation_max_length': model_data_cfg.get('max_target_length', 128) # 生成摘要的最大长度
        }
        training_args = Seq2SeqTrainingArguments(**common_args, **seq2seq_extra_args)
        compute_metrics_fn = partial(metrics_module.compute_metrics, tokenizer=tokenizer)

    else:
        training_args = TrainingArguments(**common_args)
        compute_metrics_fn = metrics_module.compute_metrics

    # --- 7. 初始化 Trainer (优雅地处理 compute_metrics) ---
    callbacks = []
    if training_cfg.get('log_distribution', False):
        print("📊 启用参数和梯度分布记录...")
        callbacks.append(DistributionLoggingCallback())

    # 根据任务类型选择 Trainer 和 DataCollator
    if task_type == 'seq2seq':
        TrainerClass = Seq2SeqTrainer
        # 为 seq2seq 任务创建 DataCollatorForSeq2Seq
        if training_cfg.get('torch_compile', False):
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                padding="max_length",
                pad_to_multiple_of=8,
                max_length=model_data_cfg.get('max_source_length', 1024),
            )
        else:
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                padding=True,
                pad_to_multiple_of=8,
            )
    else:
        TrainerClass = Trainer
        data_collator = None  # 分类任务使用默认的data collator
    
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,  # 传递data collator
        compute_metrics=compute_metrics_fn,  # 传递新创建的函数
        callbacks=callbacks
    )

    # ... [你原来的训练、评估、保存总结的逻辑完全不变] ...
    print("\n" + "=" * 40 + "\n          🔥 开始模型训练 🔥          \n" + "=" * 40 + "\n")
    
    # 如果指定了恢复训练的检查点，从检查点恢复训练状态
    if resume_from:
        print(f"🔄 从检查点恢复训练状态: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
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
        'addition': f'train from {resume_from}' if resume_from else '',
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
    parser.add_argument(
        "--resume",
        type=str,
        help="指定检查点文件夹路径，从该检查点继续训练"
    )
    args = parser.parse_args()
    main(args.task, args.resume)