import os
import math
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class DistributionLoggingCallback(TrainerCallback):
    """
    用于在TensorBoard中记录模型参数和梯度分布。
    """

    def __init__(self):
        self.tb_writer = None

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            log_dir = args.logging_dir
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=log_dir)
                print(f"DistributionLoggingCallback: TensorBoard writer initialized in '{log_dir}'.")

        if self.tb_writer is None and state.is_world_process_zero:
            print("DistributionLoggingCallback: Warning: logging_dir is not set. Distributions will not be logged.")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.tb_writer and state.is_world_process_zero and model is not None:
            print("DistributionLoggingCallback: Logging initial distribution of frozen parameters...")
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    self.tb_writer.add_histogram(
                        f'frozen_params/{name.replace(".", "/")}',
                        param.detach().cpu().numpy(),
                        global_step=0
                    )
            self.tb_writer.flush()

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """
        在每个日志记录点，只记录可训练参数的 *值*。
        """
        if self.tb_writer and state.is_world_process_zero and model is not None:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.tb_writer.add_histogram(
                        f'trainable_params/{name.replace(".", "/")}',
                        param.detach().cpu().numpy(),
                        state.global_step
                    )
            self.tb_writer.flush()

    def on_pre_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """
        在每个训练步骤结束时，捕捉并记录裁剪前的梯度。
        为了性能，我们使其与 logging_steps 的频率保持一致。
        """
        if self.tb_writer and state.is_world_process_zero and model is not None:
            # 只在日志记录步骤才执行，避免每一步都记录导致性能下降和日志文件过大
            if state.global_step > 0 and state.global_step % args.logging_steps == 0:
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self.tb_writer.add_histogram(
                            f'trainable_grads/{name.replace(".", "/")}',
                            param.grad.detach().cpu().numpy(),
                            state.global_step
                        )
                self.tb_writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.tb_writer:
            self.tb_writer.close()


class PerplexityLoggingCallback(TrainerCallback):
    """
    用于在TensorBoard中记录困惑度指标。
    """
    
    def __init__(self):
        self.tb_writer = None

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            log_dir = args.logging_dir
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=log_dir)
                print(f"PerplexityLoggingCallback: TensorBoard writer initialized in '{log_dir}'.")

        if self.tb_writer is None and state.is_world_process_zero:
            print("PerplexityLoggingCallback: Warning: logging_dir is not set. Perplexity will not be logged.")

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics and 'eval_loss' in metrics:
            eval_loss = metrics['eval_loss']
            try:
                perplexity = math.exp(eval_loss)
                metrics['eval_perplexity'] = perplexity
            except OverflowError:
                metrics['eval_perplexity'] = float('inf')

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """
        在日志记录事件中，检查是否存在评估损失（eval_loss），如果存在则计算困惑度并记录到TensorBoard。
        """
        # 使用独立的TensorBoard writer记录困惑度
        if self.tb_writer and state.is_world_process_zero and logs is not None and 'eval_loss' in logs:
            eval_loss = logs['eval_loss']
            try:
                # 计算困惑度 PPL = exp(loss)
                perplexity = math.exp(eval_loss)
                logs['eval_perplexity'] = perplexity
                # 直接记录到TensorBoard
                self.tb_writer.add_scalar('eval/perplexity', perplexity, state.global_step)
                self.tb_writer.flush()
            except OverflowError:
                logs['eval_perplexity'] = float('inf')
                self.tb_writer.add_scalar('eval/perplexity', float('inf'), state.global_step)
                self.tb_writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.tb_writer:
            self.tb_writer.close()