import cv2
import numpy as np
import pygetwindow as gt
import time
import os
from datetime import datetime
import threading
from collections import OrderedDict
from ..log.logger import Logger
import mss
class FrameManager:
    """
    单帧管理器类
    管理单帧图像及其预处理数据
    """
    
    def __init__(self, frame_image, current_time, frame_id):
        """
        初始化帧管理器
        
        参数:
            frame_image: 原始帧图像(numpy数组)
            current_time: 当前时间戳
            frame_id: 帧唯一标识符
        """
        self.frame_id = frame_id
        self.capture_time = current_time
        self.lock = threading.RLock()
        
        # 预处理图像为OpenCV需要的格式
        self.frame_data = {}
        self._preprocess_frame(frame_image)
    
    def _preprocess_frame(self, frame_image):
        """
        预处理帧图像
        将图像转换为多种OpenCV需要的格式
        """
        with self.lock:
            # 原始BGR格式
            self.frame_data['bgr'] = frame_image.copy()
            
            # 灰度图
            if len(frame_image.shape) == 3:
                self.frame_data['gray'] = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
            else:
                self.frame_data['gray'] = frame_image.copy()
            
            # RGB格式
            if len(frame_image.shape) == 3:
                self.frame_data['rgb'] = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
            
            # 图像尺寸
            self.frame_data['shape'] = frame_image.shape
            self.frame_data['height'] = frame_image.shape[0]
            self.frame_data['width'] = frame_image.shape[1] if len(frame_image.shape) > 1 else 0
            self.frame_data['channels'] = frame_image.shape[2] if len(frame_image.shape) == 3 else 1
            
            # 图像统计信息
            gray = self.frame_data['gray']
            self.frame_data['mean'] = np.mean(gray)
            self.frame_data['std'] = np.std(gray)
    
    def get(self, data_key):
        """
        获取指定格式的帧数据
        
        参数:
            data_key: 数据类型 ('bgr', 'gray', 'rgb', 'shape'等)
        
        返回:
            对应的数据，如果不存在则返回None
        """
        with self.lock:
            return self.frame_data.get(data_key, None)
    
    def delete(self):
        with self.lock:
            self.frame_data.clear()
        with self.lock:
            for key in list(self.frame_data.keys()):
                if isinstance(self.frame_data[key], np.ndarray):
                    del self.frame_data[key]
            self.frame_data.clear()


class CircularFrameBuffer:
    """
    环形帧缓冲区
    使用OrderedDict实现LRU缓存机制
    """
    
    def __init__(self, max_size=5):
        self.max_size = max_size
        self.buffer = OrderedDict()
        self.lock = threading.RLock()
        self.new_frame_event = threading.Event()  # 新帧事件，用于通知消费者
    
    def add(self, frame_mgr):
        """
        添加新帧到缓冲区
        如果缓冲区已满，自动淘汰最旧的帧
        
        返回:
            FrameManager: 被淘汰的帧（如果有），否则返回None
        """
        with self.lock:
            frame_id = frame_mgr.frame_id
            evicted_frame = None
            
            # 如果已存在相同ID的帧，先删除
            if frame_id in self.buffer:
                evicted_frame = self.buffer[frame_id]
                del self.buffer[frame_id]
            
            # 添加新帧
            self.buffer[frame_id] = frame_mgr
            
            # 如果超出最大容量，淘汰最旧的帧
            if len(self.buffer) > self.max_size:
                # popitem(last=False) 弹出最早添加的项
                oldest_id, oldest_frame = self.buffer.popitem(last=False)
                evicted_frame = oldest_frame
            
            # 设置新帧事件，唤醒等待的消费者
            self.new_frame_event.set()
            # 事件需要手动清除，这里只设置不立即清除，让消费者消费后自己清除
            
            return evicted_frame
    
    def get_latest(self):
        """
        获取最新的一帧（非阻塞）
        
        返回:
            FrameManager or None
        """
        with self.lock:
            if not self.buffer:
                return None
            # 获取最后一个（最新的）帧
            latest_id = next(reversed(self.buffer))
            return self.buffer[latest_id]
    
    def get_oldest(self):
        """
        获取最旧的一帧（非阻塞）
        
        返回:
            FrameManager or None
        """
        with self.lock:
            if not self.buffer:
                return None
            # 获取第一个（最旧的）帧
            oldest_id = next(iter(self.buffer))
            return self.buffer[oldest_id]
    
    def get_all_frames(self):
        """
        获取所有帧（按时间顺序）
        
        返回:
            list of FrameManager
        """
        with self.lock:
            return list(self.buffer.values())
    
    def wait_for_new_frame(self, timeout=None):
        """
        等待新帧到达（阻塞）
        
        参数:
            timeout: 超时时间（秒）
        
        返回:
            bool: 是否有新帧到达
        """
        # 先检查是否已有新帧
        with self.lock:
            if len(self.buffer) > 0:
                return True
        
        # 等待事件
        result = self.new_frame_event.wait(timeout)
        if result:
            # 有事件，清除它（下次需要重新等待）
            self.new_frame_event.clear()
        return result
    
    def remove(self, frame_id):
        """
        移除指定ID的帧
        
        返回:
            FrameManager or None
        """
        with self.lock:
            if frame_id in self.buffer:
                frame = self.buffer[frame_id]
                del self.buffer[frame_id]
                return frame
            return None
    
    def clear(self):
        """
        清空缓冲区
        """
        with self.lock:
            frames = list(self.buffer.values())
            self.buffer.clear()
            self.new_frame_event.clear()
            return frames
    
    def size(self):
        """
        获取当前缓冲区大小
        """
        with self.lock:
            return len(self.buffer)


class WindowExtractor:
    """
    窗口提取器类（生产者-消费者模式）
    管理窗口捕获和帧缓存
    """
    def __init__(self, log_path, fps=30, buffer_size=5):
        """
        初始化窗口提取器
        
        参数:
            log_path: 日志文件路径（必须已存在）
            fps: 目标帧率，默认30
            buffer_size: 缓冲区大小，默认5帧
        
        抛出:
            FileNotFoundError: 如果日志文件不存在
        """
        # 检查日志文件是否存在
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"日志文件不存在: {log_path}")
        
        self.log_path = log_path
        self.target_fps = fps
        self.frame_interval = 1.0 / fps
        self.buffer_size = buffer_size
        self.window = None
        self.window_name = None
        
        # 生产者相关
        self.producer_running = False
        self.producer_thread = None
        self.producer_lock = threading.RLock()
        
        # 帧ID生成
        self.special_id = 0
        self.id_lock = threading.Lock()
        
        # 环形缓冲区
        self.buffer = CircularFrameBuffer(max_size=buffer_size)
        
        # 窗口位置和尺寸参数
        self.window_left = 0
        self.window_top = 0
        self.window_width = 0
        self.window_height = 0
        
        # 初始化日志管理器（直接使用传入的log_path）
        self.logger = Logger(log_path)
        self.logger.info(f"窗口提取器初始化完成，日志文件: {log_path}, FPS: {fps}, 缓冲区大小: {buffer_size}")
        
        # 日志计数，用于控制高频日志
        self.log_counter = 0
        self.last_log_time = time.time()
        
        # 性能统计
        self.stats_lock = threading.Lock()
        self.frames_captured = 0
        self.frames_dropped = 0
        self.last_stats_time = time.time()
    def _should_log(self):
        """
        判断是否应该记录日志
        限制日志频率，每30帧左右记录一次
        """
        self.log_counter += 1
        current_time = time.time()
        
        # 每秒重置计数
        if current_time - self.last_log_time >= 1.0:
            self.log_counter = 0
            self.last_log_time = current_time
            return True
        
        # 每30帧左右记录一次
        return self.log_counter % 30 == 0
    
    def _get_next_id(self):
        """
        获取下一个帧ID（线程安全）
        """
        with self.id_lock:
            current_id = self.special_id
            self.special_id += 1
            return current_id
    
    def _update_stats(self, captured=True):
        """
        更新性能统计
        """
        with self.stats_lock:
            if captured:
                self.frames_captured += 1
            else:
                self.frames_dropped += 1
            
            current_time = time.time()
            if current_time - self.last_stats_time >= 5.0:  # 每5秒记录一次统计
                elapsed = current_time - self.last_stats_time
                capture_fps = self.frames_captured / elapsed if elapsed > 0 else 0
                drop_rate = self.frames_dropped / (self.frames_captured + self.frames_dropped + 1) * 100
                
                if self._should_log():
                    self.logger.info(f"统计: 捕获帧率={capture_fps:.2f} FPS, 丢帧率={drop_rate:.2f}%, 缓冲区大小={self.buffer.size()}")
                
                self.frames_captured = 0
                self.frames_dropped = 0
                self.last_stats_time = current_time
    
    def find(self, window_name):
        """
        查找窗口并开始跟踪
        
        参数:
            window_name: 窗口名称（支持部分匹配）
        
        返回:
            bool: 是否成功找到窗口
        """
        self.window_name = window_name
        
        try:
            windows = gt.getWindowsWithTitle(window_name)
            
            if not windows:
                if self._should_log():
                    self.logger.warning(f"未找到窗口: {window_name}")
                return False
            
            # 找到第一个匹配的窗口
            self.window = windows[0]
            
            # 检查窗口状态
            if self.window.isMinimized or not self.window.visible:
                if self._should_log():
                    self.logger.warning(f"窗口已最小化或不可见: {window_name}")
                return False
            
            self._update_window_rect()
            
            if self._should_log():
                self.logger.info(f"找到窗口: {self.window.title}, 位置: ({self.window_left}, {self.window_top}), 大小: {self.window_width}x{self.window_height}")
            
            return True
            
        except Exception as e:
            if self._should_log():
                self.logger.error(f"查找窗口失败: {e}")
            return False
    
    def _update_window_rect(self):
        """更新窗口位置和尺寸参数"""
        if self.window:
            self.window_left = self.window.left
            self.window_top = self.window.top
            self.window_width = self.window.width
            self.window_height = self.window.height
    
    def _capture_one_frame(self):
        """
        抓取一帧窗口（内部使用）
        
        返回:
            FrameManager: 新创建的帧管理器，失败返回None
        """
        if not self.window:
            return None
        
        try:
            with mss.mss() as sct:
                monitor = {
                    "left": self.window_left,
                    "top": self.window_top,
                    "width": self.window_width,
                    "height": self.window_height
                }
                
                # 捕获屏幕
                screenshot = sct.grab(monitor)
                
                # 转换为numpy数组和BGR格式
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                # 创建新帧ID
                current_id = self._get_next_id()
                
                # 创建帧管理器
                frame_mgr = FrameManager(img, time.time(), current_id)
                
                return frame_mgr
                
        except Exception as e:
            if self._should_log():
                self.logger.error(f"抓取帧失败: {e}")
            return None
    
    def _check_window_state(self):
        """
        检测窗口状态并调整抓取位置
        
        返回:
            bool: 窗口状态是否正常
        """
        if not self.window:
            return False
        
        try:
            # 重新获取窗口信息（更新位置）
            windows = gt.getWindowsWithTitle(self.window_name)
            if not windows:
                return False
            
            self.window = windows[0]
            
            # 检查窗口状态
            if self.window.isMinimized or not self.window.visible:
                return False
            
            # 检查窗口位置是否变化
            if (self.window.left != self.window_left or 
                self.window.top != self.window_top or
                self.window.width != self.window_width or
                self.window.height != self.window_height):
                
                old_rect = (self.window_left, self.window_top, self.window_width, self.window_height)
                self._update_window_rect()
                new_rect = (self.window_left, self.window_top, self.window_width, self.window_height)
                
                if self._should_log():
                    self.logger.info(f"窗口位置/大小变化: {old_rect} -> {new_rect}")
            
            return True
            
        except Exception as e:
            if self._should_log():
                self.logger.error(f"检测窗口状态失败: {e}")
            return False
    
    def _producer_loop(self):
        """
        生产者循环（在独立线程中运行）
        """
        self.logger.info("生产者线程已启动")
        
        while self.producer_running:
            loop_start = time.time()
            
            # 检测窗口状态
            if not self._check_window_state():
                if self._should_log():
                    self.logger.warning("窗口状态异常，暂停生产")
                time.sleep(0.5)  # 状态异常时等待久一点
                continue
            
            # 抓取一帧
            frame_mgr = self._capture_one_frame()
            
            if frame_mgr:
                # 添加到缓冲区
                evicted_frame = self.buffer.add(frame_mgr)
                
                # 如果有被淘汰的帧，释放其内存
                if evicted_frame:
                    evicted_frame.delete()
                    self._update_stats(captured=False)
                else:
                    self._update_stats(captured=True)
                
                if self._should_log():
                    self.logger.debug(f"生产帧 {frame_mgr.frame_id}, 当前缓冲区大小: {self.buffer.size()}")
            else:
                if self._should_log():
                    self.logger.debug("抓取帧失败")
            
            # 控制帧率
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.logger.info("生产者线程已停止")
    
    def start_capture(self):
        """
        启动生产者线程（非阻塞）
        
        返回:
            bool: 是否成功启动
        """
        with self.producer_lock:
            if self.producer_running:
                self.logger.warning("生产者已在运行")
                return False
            
            if not self.window:
                self.logger.error("未找到窗口，请先调用find()")
                return False
            
            self.producer_running = True
            self.producer_thread = threading.Thread(
                target=self._producer_loop,
                name="WindowExtractor-Producer",
                daemon=True
            )
            self.producer_thread.start()
            
            self.logger.info("生产者线程已启动")
            return True
    
    def stop_capture(self):
        """
        停止生产者线程
        """
        with self.producer_lock:
            if not self.producer_running:
                return
            
            self.producer_running = False
            self.logger.info("正在停止生产者线程...")
            
            # 等待线程结束
            if self.producer_thread and self.producer_thread.is_alive():
                self.producer_thread.join(timeout=2.0)
            
            self.logger.info("生产者线程已停止")
    
    def get_frame(self, block=False, timeout=None):
        """
        获取最新的一帧（消费者接口）
        
        参数:
            block: 是否阻塞等待新帧
            timeout: 阻塞超时时间（秒）
        
        返回:
            FrameManager or None
        """
        if block:
            # 阻塞等待新帧
            if self.buffer.wait_for_new_frame(timeout):
                return self.buffer.get_latest()
            return None
        else:
            # 非阻塞获取
            return self.buffer.get_latest()
    
    def get_all_frames(self):
        """
        获取所有缓存的帧
        
        返回:
            list of FrameManager
        """
        return self.buffer.get_all_frames()
    
    def clear_buffer(self):
        """
        清空缓冲区并释放所有帧内存
        """
        frames = self.buffer.clear()
        for frame in frames:
            frame.delete()
        self.logger.info(f"已清空缓冲区，释放 {len(frames)} 帧")
    
    def get_buffer_size(self):
        """
        获取当前缓冲区大小
        """
        return self.buffer.size()
    
    def is_running(self):
        """
        检查生产者是否在运行
        """
        return self.producer_running
    
if __name__=="__main__":
    #孩子们要记得log是存在外面的，开始一次要
    extractor = WindowExtractor(
        cache_folder="./test_cache",
        fps=30,  # 目标帧率
        buffer_size=5  # 缓冲区大小
    )
    window_names = ["记事本", "计算器", "浏览器", "无标题 - 记事本", "Untitled - Notepad"]
    found = False
    while found==False:
        for name in window_names:
            if extractor.find(name):
                found = True
    if extractor.start_capture():
        print("✅")
    else:
        print("❌ ")
        exit()
    print("\n[4] 测试缓冲区...")
    time.sleep(1)  
    buffer_size = extractor.get_buffer_size()
    print(f"当前缓冲区大小: {buffer_size}")
    print("\n[5] 测试获取帧...")
    frame = extractor.get_frame(block=False)
    if frame:
        print(f"✅ 非阻塞获取成功 - 帧ID: {frame.frame_id}")
        bgr_img = frame.get('bgr')
        gray_img = frame.get('gray')
        print(f"   图像尺寸: {frame.get('width')}x{frame.get('height')}, 通道数: {frame.get('channels')}")
        print(f"   图像均值: {frame.get('mean'):.2f}, 标准差: {frame.get('std'):.2f}")
    else:
        print("❌ 非阻塞获取失败")
    
    # 阻塞获取（带超时）
    print("\n等待新帧（阻塞2秒）...")
    frame = extractor.get_frame(block=True, timeout=2)
    if frame:
        print(f"✅ 阻塞获取成功 - 帧ID: {frame.frame_id}")
    else:
        print("❌ 阻塞获取超时")
    
    # 6. 测试获取所有帧
    print("\n[6] 测试获取所有帧...")
    all_frames = extractor.get_all_frames()
    print(f"当前缓冲区内有 {len(all_frames)} 帧")
    for i, f in enumerate(all_frames):
        print(f"  帧[{i}]: ID={f.frame_id}, 时间={f.capture_time:.3f}")
    
    # 7. 测试连续获取（模拟消费者）
    print("\n[7] 模拟消费者连续获取5次...")
    for i in range(5):
        frame = extractor.get_frame()
        if frame:
            print(f"  第{i+1}次: 获取到帧 ID={frame.frame_id}")
        else:
            print(f"  第{i+1}次: 无帧")
        time.sleep(0.1)
    
    # 8. 测试FPS稳定性
    print("\n[8] 测试FPS稳定性（3秒采样）...")
    start_time = time.time()
    frame_times = []
    frame_ids = []
    
    while time.time() - start_time < 3:
        frame = extractor.get_frame()
        if frame:
            current_time = time.time()
            frame_times.append(current_time)
            frame_ids.append(frame.frame_id)
        time.sleep(0.001)
    
    if len(frame_times) > 1:
        intervals = []
        for i in range(1, len(frame_times)):
            interval = frame_times[i] - frame_times[i-1]
            intervals.append(interval)
        
        avg_interval = sum(intervals) / len(intervals)
        avg_fps = 1.0 / avg_interval if avg_interval > 0 else 0
        std_dev = np.std(intervals) if intervals else 0
        
        print(f"  采样帧数: {len(frame_times)}")
        print(f"  平均帧间隔: {avg_interval*1000:.2f}ms")
        print(f"  实际FPS: {avg_fps:.2f} (目标: {extractor.target_fps})")
        print(f"  间隔标准差: {std_dev*1000:.2f}ms")
        print(f"  帧ID范围: {min(frame_ids)} -> {max(frame_ids)}")
    
    # 9. 测试清空缓冲区
    print("\n[9] 测试清空缓冲区...")
    before_clear = extractor.get_buffer_size()
    extractor.clear_buffer()
    after_clear = extractor.get_buffer_size()
    print(f"  清空前: {before_clear}帧, 清空后: {after_clear}帧")
    
    # 10. 测试窗口移动/大小调整
    print("\n[10] 提示: 请尝试移动窗口或调整窗口大小")
    print("   5秒后将自动检测变化...")
    time.sleep(5)
    
    # 重新获取一帧查看效果
    frame = extractor.get_frame()
    if frame:
        print(f"  当前帧 ID: {frame.frame_id}")
        print(f"  窗口位置: ({extractor.window_left}, {extractor.window_top})")
        print(f"  窗口大小: {extractor.window_width}x{extractor.window_height}")
    
    # 11. 测试停止捕获
    print("\n[11] 停止捕获线程...")
    extractor.stop_capture()
    time.sleep(1)
    print(f"  生产者运行状态: {extractor.is_running()}")
    
    # 12. 测试重新启动
    print("\n[12] 测试重新启动捕获...")
    if extractor.start_capture():
        print("✅ 重新启动成功")
        time.sleep(1)
        frame = extractor.get_frame()
        if frame:
            print(f"  重新启动后获取到帧 ID: {frame.frame_id}")
    else:
        print("❌ 重新启动失败")
    
    # 13. 最终清理
    print("\n[13] 最终清理...")
    extractor.stop_capture()
    extractor.clear_buffer()


if __name__ == "__main__":
    print("="*60)
    print("窗口提取器测试程序")
    print("="*60)
    
    # 1. 先创建日志文件
    print("\n[1] 创建日志文件...")
    log_path = "./test_window_extractor.log"
    
    # 确保日志文件存在
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"# 窗口提取器日志 - 创建时间: {datetime.now()}\n")
        print(f"✅ 已创建日志文件: {log_path}")
    except Exception as e:
        print(f"❌ 创建日志文件失败: {e}")
        exit()
    
    # 2. 初始化提取器
    print("\n[2] 初始化窗口提取器...")
    try:
        extractor = WindowExtractor(
            log_path=log_path,  # 直接传入日志文件路径
            fps=30,
            buffer_size=5
        )
        print("✅ 窗口提取器初始化成功")
    except FileNotFoundError as e:
        print(f"❌ 初始化失败: {e}")
        exit()
    
    # 3. 查找窗口
    print("\n[3] 查找窗口...")
    print("请确保已打开一个窗口（如记事本、浏览器或计算器）")
    
    # 尝试常见的窗口名称
    window_names = ["记事本", "计算器", "浏览器", "无标题 - 记事本", "Untitled - Notepad", "微信", "QQ"]
    found = False
    
    for name in window_names:
        print(f"   尝试查找: {name}")
        if extractor.find(name):
            print(f"   ✅ 成功找到窗口: {name}")
            found = True
            break
    
    if not found:
        # 手动输入窗口名称
        custom_name = input("   请输入窗口名称（部分匹配）: ")
        if extractor.find(custom_name):
            print(f"   ✅ 成功找到窗口: {custom_name}")
            found = True
    
    if not found:
        print("   ❌ 未找到任何窗口，退出测试")
        exit()
    
    # 4. 启动捕获
    print("\n[4] 启动捕获线程...")
    if extractor.start_capture():
        print("   ✅ 捕获线程已启动")
    else:
        print("   ❌ 启动失败")
        exit()
    
    # 5. 测试缓冲区状态
    print("\n[5] 测试缓冲区...")
    time.sleep(1)  # 等待捕获几帧
    
    buffer_size = extractor.get_buffer_size()
    print(f"   当前缓冲区大小: {buffer_size}")
    
    # 6. 测试获取帧
    print("\n[6] 测试获取帧...")
    
    # 非阻塞获取
    frame = extractor.get_frame(block=False)
    if frame:
        print(f"   ✅ 非阻塞获取成功 - 帧ID: {frame.frame_id}")
        bgr_img = frame.get('bgr')
        gray_img = frame.get('gray')
        print(f"      图像尺寸: {frame.get('width')}x{frame.get('height')}, 通道数: {frame.get('channels')}")
        print(f"      图像均值: {frame.get('mean'):.2f}, 标准差: {frame.get('std'):.2f}")
    else:
        print("   ❌ 非阻塞获取失败")
    
    # 阻塞获取（带超时）
    print("\n   等待新帧（阻塞2秒）...")
    frame = extractor.get_frame(block=True, timeout=2)
    if frame:
        print(f"   ✅ 阻塞获取成功 - 帧ID: {frame.frame_id}")
    else:
        print("   ❌ 阻塞获取超时")
    
    # 7. 测试获取所有帧
    print("\n[7] 测试获取所有帧...")
    all_frames = extractor.get_all_frames()
    print(f"   当前缓冲区内有 {len(all_frames)} 帧")
    for i, f in enumerate(all_frames):
        print(f"      帧[{i}]: ID={f.frame_id}, 时间={f.capture_time:.3f}")
    
    # 8. 测试连续获取（模拟消费者）
    print("\n[8] 模拟消费者连续获取5次...")
    for i in range(5):
        frame = extractor.get_frame()
        if frame:
            print(f"      第{i+1}次: 获取到帧 ID={frame.frame_id}")
        else:
            print(f"      第{i+1}次: 无帧")
        time.sleep(0.1)
    
    # 9. 测试FPS稳定性
    print("\n[9] 测试FPS稳定性（3秒采样）...")
    start_time = time.time()
    frame_times = []
    frame_ids = []
    
    while time.time() - start_time < 3:
        frame = extractor.get_frame()
        if frame:
            current_time = time.time()
            frame_times.append(current_time)
            frame_ids.append(frame.frame_id)
        time.sleep(0.001)
    
    if len(frame_times) > 1:
        intervals = []
        for i in range(1, len(frame_times)):
            interval = frame_times[i] - frame_times[i-1]
            intervals.append(interval)
        
        avg_interval = sum(intervals) / len(intervals)
        avg_fps = 1.0 / avg_interval if avg_interval > 0 else 0
        std_dev = np.std(intervals) if intervals else 0
        
        print(f"      采样帧数: {len(frame_times)}")
        print(f"      平均帧间隔: {avg_interval*1000:.2f}ms")
        print(f"      实际FPS: {avg_fps:.2f} (目标: {extractor.target_fps})")
        print(f"      间隔标准差: {std_dev*1000:.2f}ms")
        print(f"      帧ID范围: {min(frame_ids)} -> {max(frame_ids)}")
    
    # 10. 测试清空缓冲区
    print("\n[10] 测试清空缓冲区...")
    before_clear = extractor.get_buffer_size()
    extractor.clear_buffer()
    after_clear = extractor.get_buffer_size()
    print(f"      清空前: {before_clear}帧, 清空后: {after_clear}帧")
    
    # 11. 测试窗口移动/大小调整
    print("\n[11] 提示: 请尝试移动窗口或调整窗口大小")
    print("      5秒后将自动检测变化...")
    time.sleep(5)
    
    # 重新获取一帧查看效果
    frame = extractor.get_frame()
    if frame:
        print(f"      当前帧 ID: {frame.frame_id}")
        print(f"      窗口位置: ({extractor.window_left}, {extractor.window_top})")
        print(f"      窗口大小: {extractor.window_width}x{extractor.window_height}")
    
    # 12. 测试停止捕获
    print("\n[12] 停止捕获线程...")
    extractor.stop_capture()
    time.sleep(1)
    print(f"      生产者运行状态: {extractor.is_running()}")
    
    # 13. 测试重新启动
    print("\n[13] 测试重新启动捕获...")
    if extractor.start_capture():
        print("      ✅ 重新启动成功")
        time.sleep(1)
        frame = extractor.get_frame()
        if frame:
            print(f"      重新启动后获取到帧 ID: {frame.frame_id}")
    else:
        print("      ❌ 重新启动失败")
    
    # 14. 最终清理
    print("\n[14] 最终清理...")
    extractor.stop_capture()
    extractor.clear_buffer()
    print("   ✅ 测试完成")
    
    print("\n" + "="*60)
    print(f"测试结束，请查看日志文件: {log_path}")
    print("="*60)