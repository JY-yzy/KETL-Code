import os
import sys
import logging
from datetime import datetime
from typing import Optional

class ExperimentLogger:
    """实验日志记录器"""
    
    def __init__(self, experiment_name: str = None, log_dir: str = "logs"):
        """
        Args:
            experiment_name: 实验名称，用于创建日志文件夹
            log_dir: 日志根目录
        """
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = os.path.join(log_dir, self.experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.DEBUG)
        
        # 移除已存在的处理器，避免重复输出
        self.logger.handlers.clear()
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器（DEBUG级别）
        debug_file = os.path.join(self.log_dir, 'debug.log')
        file_handler = logging.FileHandler(debug_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # 结果文件（用于记录关键指标）
        self.results_file = os.path.join(self.log_dir, 'results.log')
        
        self.start_time = datetime.now()
        self.log_params = {}
        
        self.info("="*60)
        self.info(f"实验开始: {self.experiment_name}")
        self.info(f"开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.info("="*60)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def critical(self, message: str):
        self.logger.critical(message)
    
    def log_params(self, params: dict):
        """记录实验参数"""
        self.log_params.update(params)
        self.info("实验参数:")
        for key, value in params.items():
            self.info(f"  {key}: {value}")
    
    def log_metrics(self, epoch: int, metrics: dict):
        """记录训练指标"""
        self.info(f"Epoch {epoch} 指标:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.6f}")
            else:
                self.info(f"  {key}: {value}")
        
        # 同时写入结果文件
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write(f"{epoch}\t{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t")
            f.write("\t".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))
            f.write("\n")
    
    def log_evaluation(self, dataset_name: str, metrics: dict):
        """记录评估结果"""
        self.info(f"数据集 {dataset_name} 评估结果:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.4f}")
            else:
                self.info(f"  {key}: {value}")
        
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write(f"EVAL\t{dataset_name}\t")
            f.write("\t".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))
            f.write("\n")
    
    def log_stage(self, stage_name: str, stage_desc: str):
        """记录阶段开始"""
        self.info("-"*60)
        self.info(f"阶段: {stage_name}")
        self.info(f"描述: {stage_desc}")
        self.info("-"*60)
    
    def log_ablation_result(self, model_name: str, metrics: dict):
        """记录消融实验结果"""
        self.info(f"消融实验 [{model_name}]:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.4f}")
            else:
                self.info(f"  {key}: {value}")
    
    def end_experiment(self):
        """结束实验并记录总结"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        self.info("="*60)
        self.info(f"实验结束: {self.experiment_name}")
        self.info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.info(f"持续时间: {duration.total_seconds():.2f} 秒")
        self.info("="*60)
        
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write(f"END\t{end_time.strftime('%Y-%m-%d %H:%M:%S')}\t{duration.total_seconds():.2f}\n")
    
    def get_log_dir(self) -> str:
        """获取日志目录路径"""
        return self.log_dir


class TrainingProgressLogger:
    """训练进度日志记录器"""
    
    def __init__(self, logger: ExperimentLogger):
        self.logger = logger
        self.epoch_start_time = None
    
    def start_epoch(self, epoch: int, total_epochs: int):
        """开始一个epoch"""
        self.epoch_start_time = datetime.now()
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Epoch {epoch}/{total_epochs}")
        self.logger.info(f"{'='*60}")
    
    def log_batch(self, batch_idx: int, total_batches: int, loss: float):
        """记录batch信息"""
        if batch_idx % max(1, total_batches // 10) == 0:
            progress = (batch_idx / total_batches) * 100
            self.logger.debug(f"Batch {batch_idx}/{total_batches} ({progress:.1f}%) - Loss: {loss:.6f}")
    
    def end_epoch(self, epoch: int, train_loss: float, val_metrics: dict = None):
        """结束一个epoch"""
        epoch_time = datetime.now() - self.epoch_start_time
        self.logger.info(f"训练损失: {train_loss:.6f}")
        self.logger.info(f"Epoch耗时: {epoch_time.total_seconds():.2f} 秒")
        
        if val_metrics:
            self.logger.info("验证集指标:")
            for key, value in val_metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"  {key}: {value:.4f}")
                else:
                    self.logger.info(f"  {key}: {value}")
    
    def log_learning_rate(self, lr: float):
        """记录学习率"""
        self.logger.debug(f"学习率: {lr:.6e}")


def get_logger(experiment_name: str = None) -> ExperimentLogger:
    """获取日志记录器单例"""
    return ExperimentLogger(experiment_name)


if __name__ == "__main__":
    # 测试日志模块
    logger = get_logger("test_experiment")
    logger.log_params({"batch_size": 32, "lr": 3e-5, "epochs": 10})
    logger.log_metrics(1, {"loss": 0.5, "accuracy": 0.85})
    logger.log_evaluation("test_set", {"f1": 0.82, "precision": 0.81, "recall": 0.83})
    logger.end_experiment()
