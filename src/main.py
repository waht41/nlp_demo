import os
import yaml
import shutil
import argparse
import pandas as pd
import torch
from datetime import datetime
import importlib
from functools import partial

from transformers import (
    TrainingArguments, Trainer,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoModelForCausalLM,
    DataCollatorForSeq2Seq, DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from trainer_callback import DistributionLoggingCallback, PerplexityLoggingCallback
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
    config_path = os.path.join(TASKS_DIR, task_name, "config.yaml")

    print(f"📖 从 '{config_path}' 加载配置...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    metadata_cfg = config.get('metadata', {})
    model_data_cfg = config.get('model_data', {})
    training_cfg = config.get('training', {})

    task_type = model_data_cfg.get('task_type', 'classification')
    if task_type not in ['classification', 'seq2seq', 'causalLM']:
        raise ValueError(f"未知的任务类型: {task_type}")

    print(f"检测到任务类型: {task_type}")

    try:
        # 动态导入特定任务的数据处理模块（必需）
        data_handler_module = importlib.import_module(f"tasks.{task_name}.data_handler")
        
        # 根据配置决定是否需要导入metrics模块
        ignore_metrics = training_cfg.get('ignore_compute_metric', False)
        metrics_module = None if ignore_metrics else importlib.import_module(f"tasks.{task_name}.metrics")
        if ignore_metrics:
            print("📊 跳过metrics模块导入（根据配置ignore_compute_metric=true）")
    except ModuleNotFoundError as e:
        print(f"错误: 导入任务 '{task_name}' 相关模块时失败。")
        print(f"具体错误: {str(e)}")
        print("可能的原因:")
        print("1. tasks/ 目录下没有对应的任务文件夹")
        print("2. 任务文件夹中缺少必需的 data_handler.py 或 metrics.py 文件")
        print("3. 任务模块中引用的依赖包未安装，请检查 requirements.txt 并安装所需依赖")
        return

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

    # --- 4. 加载数据集 ---
    print("\n" + "=" * 20 + " 正在加载数据集 " + "=" * 20)
    tokenizer = AutoTokenizer.from_pretrained(model_data_cfg['model_checkpoint'], trust_remote_code=True)

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
            max_source_length=model_data_cfg.get('max_source_length'),
            max_target_length=model_data_cfg.get('max_target_length')
        )
    elif task_type == 'causalLM':
        # 为 causalLM 任务设置 pad_token
        # 确保 tokenizer 有 eos_token
        if tokenizer.eos_token is None:
            tokenizer.eos_token = "<|endoftext|>"
            print('添加eos_token <|endoftext|>')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        datasets_and_labels = data_handler_module.load_and_prepare_dataset(
            **common_dataset_args,
            max_length=model_data_cfg.get('max_length', 1024)
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

    # 在 causalLM 逻辑块外部定义 lora_config，以便 Trainer 部分可以访问
    lora_config = None

    if task_type == 'seq2seq':
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    elif task_type == 'causalLM':
        # --- LLM 高效微调的核心逻辑 ---
        quantization_cfg = training_cfg.get('quantization')
        bnb_config = None
        if quantization_cfg and quantization_cfg.get('load_in_4bit', False):
            print("💡 启用 4-bit 量化加载...")
            compute_dtype = getattr(torch, quantization_cfg.get('bnb_4bit_compute_dtype', 'float16'))
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=quantization_cfg.get('bnb_4bit_use_double_quant', True),
                bnb_4bit_quant_type=quantization_cfg.get('bnb_4bit_quant_type', "nf4"),
                bnb_4bit_compute_dtype=compute_dtype
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        if training_cfg.get('use_peft', False):
            print("🚀 应用 PEFT (LoRA) 配置...")
            model = prepare_model_for_kbit_training(model)
            peft_lora_cfg = training_cfg.get('peft_lora')
            if not peft_lora_cfg:
                raise ValueError("配置错误: use_peft=true 但 peft_lora 配置块不存在！")
            lora_config = LoraConfig(**peft_lora_cfg)
            model = get_peft_model(model, lora_config)
            print("LoRA 模型参数:")
            model.print_trainable_parameters()
    else:  # 默认为 classification
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

    if model_data_cfg.get('force_contiguous', False):
        model = model.to_contiguous()

    # --- 6. 配置训练参数 ---
    print("\n" + "=" * 20 + " 正在配置训练参数 " + "=" * 20)
    common_args = {
        'output_dir': OUTPUT_DIR, 'logging_dir': LOGGING_DIR, 'report_to': "tensorboard",
        'run_name': run_id, 'optim': training_cfg.get('optimizer', 'adamw_torch'),
        'gradient_accumulation_steps': training_cfg.get('gradient_accumulation_steps', 1),
        'num_train_epochs': training_cfg['num_train_epochs'],
        'per_device_train_batch_size': training_cfg['per_device_train_batch_size'],
        'per_device_eval_batch_size': training_cfg.get('per_device_eval_batch_size',
                                                       training_cfg['per_device_train_batch_size']),
        'learning_rate': float(training_cfg['learning_rate']),
        'weight_decay': training_cfg.get('weight_decay', 0.0),
        'max_grad_norm': training_cfg.get('max_grad_norm', 1.0),
        'warmup_ratio': training_cfg.get('warmup_ratio', 0.0),
        'lr_scheduler_type': training_cfg.get('lr_scheduler_type', 'linear'),
        'eval_strategy': training_cfg.get('eval_strategy', 'epoch'),
        'save_strategy': training_cfg.get('save_strategy', 'epoch'),
        'eval_steps': training_cfg.get('eval_steps') if training_cfg.get('eval_strategy') == 'steps' else None,
        'save_steps': training_cfg.get('save_steps') if training_cfg.get('save_strategy') == 'steps' else None,
        'load_best_model_at_end': True,
        'logging_strategy': "steps",
        'logging_steps': training_cfg.get('logging_steps', 50),
        'fp16': training_cfg.get('fp16', False),
        'bf16': training_cfg.get('bf16', False),
        'torch_compile': training_cfg.get('torch_compile', False),
    }
    compute_metrics_fn = partial(metrics_module.compute_metrics,
                                 tokenizer=tokenizer) if metrics_module and task_type == 'seq2seq' else \
        metrics_module.compute_metrics if metrics_module else None

    if task_type == 'seq2seq':
        training_args = Seq2SeqTrainingArguments(**common_args, predict_with_generate=True,
                                                 generation_max_length=model_data_cfg.get('max_target_length', 128))
    else:
        training_args = TrainingArguments(**common_args)

    # --- 7. 初始化 Trainer ---
    callbacks = []
    if training_cfg.get('log_distribution'):
        callbacks.append(DistributionLoggingCallback())
    if training_cfg.get('log_ppl'):
        callbacks.append(PerplexityLoggingCallback())

    if task_type == 'seq2seq':
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)
        trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=train_dataset,
                                 eval_dataset=eval_dataset,
                                 tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics_fn,
                                 callbacks=callbacks)
    elif task_type == 'causalLM':
        if training_cfg.get('use_peft', False):
            print("使用 TRL 的 SFTTrainer 进行 PEFT 微调...")
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                peft_config=lora_config,
                data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8),
                callbacks=callbacks,
            )
        else:
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)
            trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset,
                              tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics_fn,
                              callbacks=callbacks)
    else:  # classification
        trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset,
                          compute_metrics=compute_metrics_fn, callbacks=callbacks)

    # --- 8. 开始训练 ---
    print("\n" + "=" * 40 + "\n          🔥 开始模型训练 🔥          \n" + "=" * 40 + "\n")
    train_kwargs = {'resume_from_checkpoint': resume_from} if resume_from else {}
    trainer.train(**train_kwargs)
    print("\n" + "=" * 40 + "\n          ✅ 训练完成 ✅          \n" + "=" * 40 + "\n")

    # --- 9. 最终评估和日志记录 ---
    print("在最终评估集上进行评估...")
    final_metrics = trainer.evaluate(eval_dataset)
    print("最终评估结果:", final_metrics)

    summary = {
        'task': task_name,
        'run_id': run_id,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'run_name': metadata_cfg.get('run_name', ''),
        'description': metadata_cfg.get('description', ''),
        'git_hash': git_hash,
        'model_checkpoint': model_data_cfg['model_checkpoint'],
        'dataset_name': model_data_cfg['dataset_name'],
        'learning_rate': training_cfg.get('learning_rate'),
        'epochs': training_cfg['num_train_epochs'],
        'batch_size': training_cfg['per_device_train_batch_size'],
        'final_eval_accuracy': final_metrics.get('eval_accuracy'),
        'final_eval_loss': final_metrics.get('eval_loss'),
        'results_path': OUTPUT_DIR,
        'addition': f'train from {resume_from}' if resume_from else '',
    }
    log_file = "./experiments.csv"
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False, encoding='utf-8-sig')
    print("\n" + "=" * 40)
    print(f"   📊 实验总结已记录到中央日志: {log_file}   ")
    print("=" * 40 + "\n")

    # --- 10. 保存最终模型 ---
    final_model_path = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(final_model_path)
    if training_cfg.get('use_peft', False):  # 如果是 LoRA 训练，额外保存 tokenizer
        tokenizer.save_pretrained(final_model_path)
    print(f"最佳模型（或适配器）已保存至: {final_model_path}")
    print(f"要查看训练日志，请在终端运行: tensorboard --logdir ./logs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从指定的任务目录运行模型训练。")
    parser.add_argument(
        "--task", type=str, required=True,
        help="要执行的任务名称 (必须是 tasks/ 目录下的一个子文件夹名)"
    )
    parser.add_argument(
        "--resume", type=str,
        help="指定检查点文件夹路径，从该检查点继续训练"
    )
    args = parser.parse_args()
    main(args.task, args.resume)