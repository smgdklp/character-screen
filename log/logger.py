import threading
import os
from datetime import datetime
class Logger:
    '''用类log处理器管理器，要求初始要定义指向一个log，
     然后四项基本操作全部追加，没有则报错。
     会作为多个不同功能管理器分发，创立多个并行，共同填写同一个log，
    所以要做好数据锁。'''
    def __init__(self, log_path):
        self.log_path = log_path
        self.lock = threading.Lock()
        # 检查日志文件是否存在
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"日志文件不存在: {log_path}")
    def _write_log(self, level, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        thread_name = threading.current_thread().name
        log_entry = f"[{timestamp}] [{level}] [{thread_name}] {message}\n"
        
        with self.lock:  # 使用锁保证线程安全
            try:
                with open(self.log_path, 'a', encoding='utf-8') as f:
                    f.write(log_entry)
            except Exception as e:
                raise IOError(f"写入日志失败: {e}")
    
    def info(self, message):
        self._write_log("INFO", message)
    
    def warning(self, message):
        self._write_log("WARNING", message)
    
    def error(self, message):
        self._write_log("ERROR", message)
    
    def debug(self, message):
        self._write_log("DEBUG", message)
