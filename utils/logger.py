import sys
import time
from pathlib import Path
import yaml
from configs.config import cfg_to_dict
class Logger:
    def __init__(self, file_path):
        self.console = sys.stdout
        self.file = open(file_path, 'w', encoding='utf-8')
    def write(self, message):
        self.console.write(message)
        self.file.write(message)
        self.file.flush()
    def flush(self):
        self.console.flush()
        self.file.flush()

def setup_logger(cfg,log_dir="logs"):
    # 创建logs目录（不存在则创建）
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    # 日志文件名（按时间命名，避免覆盖）
    log_file = log_path / f"train_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    # 重定向stdout到文件（同时打印到控制台和文件）
    sys.stdout = Logger(log_file)
    print(f"[INFO] Log will be saved to {log_file}")
    print("\n===== CONFIG =====")
    print(yaml.dump(cfg_to_dict(cfg), sort_keys=False))
    print("==================\n")