import av
import cv2
import numpy as np
import base64
import time
import threading
from flask import Flask, render_template_string, request
from flask_socketio import SocketIO
import traceback

# 配置参数
rtmp_url = 'rtmp://192.168.1.2:1935/live/123'
app = Flask(__name__)

# 优化Socket.IO配置，增强连接稳定性
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    ping_timeout=60,       # 增加心跳超时时间到60秒
    ping_interval=25,      # 调整心跳间隔到25秒
    transports=['websocket', 'polling'],  # 优先使用websocket，降级到polling
    max_http_buffer_size=1e8,  # 增加最大缓冲区大小到100MB，处理大型视频帧
    async_mode='threading'  # 使用线程模式，适合I/O密集型应用
)

# 音视频流处理线程
class StreamProcessor:
    def __init__(self):
        self.container = None
        self.video_stream = None
        self.audio_stream = None
        self.running = False
        self.lock = threading.Lock()
        self.last_video_time = 0
        self.last_audio_time = 0
        self.video_frame_count = 0
        self.audio_frame_count = 0
        self.start_time = 0
        self.is_stream_alive = False
    
    def start(self):
        self.running = True
        self.start_time = time.time()
        self.video_frame_count = 0
        self.audio_frame_count = 0
        threading.Thread(target=self._process_stream, daemon=True).start()
        threading.Thread(target=self._send_heartbeat, daemon=True).start()
        threading.Thread(target=self._monitor_stream, daemon=True).start()
    
    def stop(self):
        self.running = False
        if self.container:
            self.container.close()
    
    def _send_heartbeat(self):
        """定期发送心跳包，保持连接"""
        while self.running:
            try:
                socketio.emit('heartbeat', {
                    'timestamp': time.time(),
                    'stream_alive': self.is_stream_alive,
                    'video_frame_count': self.video_frame_count,
                    'audio_frame_count': self.audio_frame_count
                })
                socketio.sleep(5)
            except Exception as e:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] 发送心跳包时出错: {e}")
                socketio.sleep(1)  # 发生错误时减少等待时间，快速重试
    
    def _monitor_stream(self):
        """监控流状态，检查是否有帧处理"""
        while self.running:
            time.sleep(10)
            current_time = time.time()
            
            # 检查视频流是否活跃（最近10秒内有帧处理）
            video_active = (current_time - self.last_video_time) < 10
            audio_active = (current_time - self.last_audio_time) < 10
            
            self.is_stream_alive = video_active or audio_active
            
            if not self.is_stream_alive:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 警告：流可能已中断，最近视频帧时间: {self.last_video_time:.2f}, 最近音频帧时间: {self.last_audio_time:.2f}")
    
    def _process_stream(self):
        """处理RTMP流，提取音视频帧并发送到前端"""
        retry_count = 0
        max_retries = 10  # 增加重试次数到10次
        
        while self.running and retry_count < max_retries:
            try:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] 尝试打开RTMP流 (尝试 {retry_count+1}/{max_retries}): {rtmp_url}")
                
                # 使用PyAV打开RTMP流，简化选项以提高稳定性
                self.container = av.open(rtmp_url, mode='r')
                print(f"[{timestamp}] 容器打开成功")
                
                # 获取音视频流
                self.video_stream = None
                self.audio_stream = None
                
                for stream in self.container.streams:
                    if stream.type == 'video':
                        self.video_stream = stream
                        print(f"[{timestamp}] 视频流: index={stream.index}, codec={stream.codec_context.name}, framerate={stream.average_rate}")
                    elif stream.type == 'audio':
                        self.audio_stream = stream
                        print(f"[{timestamp}] 音频流: index={stream.index}, codec={stream.codec_context.name}, sample_rate={stream.codec_context.sample_rate}")
                
                if not self.video_stream:
                    print(f"[{timestamp}] 未找到视频流")
                    time.sleep(2)
                    retry_count += 1
                    continue
                    
                # 重置重试计数
                retry_count = 0
                print(f"[{timestamp}] 开始处理音视频流")
                
                # 解码音视频帧
                for packet in self.container.demux():
                    if not self.running:
                        print(f"[{timestamp}] 停止处理流")
                        break
                        
                    try:
                        # 处理音视频流
                        if packet.stream.type == 'video':
                            for frame in packet.decode():
                                self._process_video_frame(frame)
                        elif packet.stream.type == 'audio':
                            for frame in packet.decode():
                                self._process_audio_frame(frame)
                        
                    except Exception as e:
                        print(f"[{timestamp}] 解码帧时出错: {e}")
                        traceback.print_exc()
                        continue
                        
            except av.error.HTTPError as e:
                print(f"[{timestamp}] HTTP错误: {e}，状态码: {getattr(e, 'errno', '未知')}")
                retry_count += 1
                time.sleep(3)
            except av.error.ConnectionError as e:
                print(f"[{timestamp}] 连接错误: {e}")
                retry_count += 1
                time.sleep(5)  # 增加连接错误的重试间隔
            except av.error.FFmpegError as e:
                print(f"[{timestamp}] FFmpeg错误: {e}")
                retry_count += 1
                time.sleep(3)
            except Exception as e:
                print(f"[{timestamp}] 处理流时出错: {type(e).__name__}: {e}")
                print(f"[{timestamp}] 错误详情: {traceback.format_exc()}")
                retry_count += 1
                time.sleep(3)
            finally:
                if self.container:
                    try:
                        self.container.close()
                        print(f"[{timestamp}] 容器已关闭")
                    except Exception as e:
                        print(f"[{timestamp}] 关闭容器时出错: {e}")
        
        if retry_count >= max_retries:
            print(f"[{timestamp}] 打开RTMP流失败，已达到最大重试次数 ({max_retries})")
    
    def _process_video_frame(self, frame):
        """处理视频帧，转换为base64并发送"""
        try:
            start_time = time.time()
            self.video_frame_count += 1
            self.is_stream_alive = True
            
            # 将PyAV帧转换为numpy数组
            img = frame.to_ndarray(format='bgr24')
            
            # 调整大小以减少传输数据量
            img_resized = cv2.resize(img, (1920, 1080))
            
            # 编码为JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]
            result, encimg = cv2.imencode('.jpg', img_resized, encode_param)
            if not result:
                return
                
            # 转换为base64
            img_base64 = base64.b64encode(encimg.tobytes()).decode('utf-8')
            
            # 获取时间戳
            timestamp = frame.pts * frame.time_base.denominator / frame.time_base.numerator if frame.pts is not None else time.time()
            
            # 发送到前端
            socketio.emit('video_frame', {
                'data': img_base64,
                'timestamp': timestamp,
                'width': 1920,
                'height': 1080
            })
            
            self.last_video_time = timestamp
            
            # 每100帧打印一次处理信息
            if self.video_frame_count % 100 == 0:
                process_time = time.time() - start_time
                fps = self.video_frame_count / (time.time() - self.start_time)
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 已处理视频帧: {self.video_frame_count}, FPS: {fps:.2f}, 单帧处理时间: {process_time*1000:.2f}ms")
            
        except Exception as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 处理视频帧时出错: {e}")
            traceback.print_exc()
    
    def _process_audio_frame(self, frame):
        """处理音频帧，转换为base64并发送"""
        try:
            self.audio_frame_count += 1
            self.is_stream_alive = True
            
            # 将PyAV音频帧转换为numpy数组
            audio_data = frame.to_ndarray()
            
            # 将float32音频数据转换为int16
            if audio_data.dtype == np.float32:
                audio_data = (audio_data * 32767).astype(np.int16)
            
            # 转换为base64
            audio_base64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
            
            # 获取时间戳
            timestamp = frame.pts * frame.time_base.denominator / frame.time_base.numerator if frame.pts is not None else time.time()
            
            # 获取音频通道数 - 使用更可靠的方式
            channels = frame.ch_layout.channels if hasattr(frame, 'ch_layout') else 2  # 默认2通道
            
            # 发送到前端
            socketio.emit('audio_frame', {
                'data': audio_base64,
                'timestamp': timestamp,
                'sample_rate': frame.sample_rate,
                'channels': channels,
                'format': 'int16' if audio_data.dtype == np.int16 else str(audio_data.dtype)
            })
            
            self.last_audio_time = timestamp
            
            # 每500帧打印一次处理信息
            if self.audio_frame_count % 500 == 0:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 已处理音频帧: {self.audio_frame_count}")
            
        except Exception as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 处理音频帧时出错: {e}")
            traceback.print_exc()

# 初始化流处理器
stream_processor = StreamProcessor()

# Flask路由
@app.route('/')
def index():
    html = '''
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>音画同步直播</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                background-color: #1a1a1a;
                color: white;
                margin: 0;
                padding: 20px;
            }
            h1 {
                color: #4CAF50;
            }
            .stream-container {
                margin: 20px auto;
                max-width: 800px;
            }
            #videoCanvas {
                border: 2px solid #4CAF50;
                background-color: #000;
                max-width: 100%;
                height: auto;
            }
            .controls {
                margin-top: 20px;
                padding: 15px;
                background-color: #333;
                border-radius: 8px;
                display: inline-block;
            }
            button {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 4px;
                transition: background-color 0.3s;
            }
            button:hover {
                background-color: #45a049;
            }
            button:disabled {
                background-color: #cccccc;
                cursor: not-allowed;
            }
            .status {
                margin-top: 15px;
                padding: 10px;
                background-color: #555;
                border-radius: 4px;
                font-size: 14px;
            }
            .sync-info {
                color: #4CAF50;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <h1>音画同步直播</h1>
        
        <div class="stream-container">
            <canvas id="videoCanvas" width="640" height="480"></canvas>
        </div>
        
        <div class="controls">
            <button id="playBtn" onclick="togglePlay()">播放</button>
            <button id="syncBtn" onclick="syncAudioVideo()">重置流</button>
        </div>
        
        <div class="status">
            <div>连接状态: <span id="connectionStatus">未连接</span></div>
            <div>视频帧率: <span id="videoFps">0</span> fps</div>
            <div>音画延迟: <span id="delayDisplay">0.00</span> ms</div>
            <div>当前状态: <span id="syncStatus" class="sync-info">初始化中...</span></div>
        </div>
        
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.0/socket.io.min.js"></script>
        <script>
            // 全局变量
            const canvas = document.getElementById('videoCanvas');
            const ctx = canvas.getContext('2d');
            const playBtn = document.getElementById('playBtn');
            const connectionStatus = document.getElementById('connectionStatus');
            const videoFps = document.getElementById('videoFps');
            const syncStatus = document.getElementById('syncStatus');
            const delayDisplay = document.getElementById('delayDisplay');
            
            let audioContext = null;
            let isPlaying = false;
            let audioDelay = 0; // 音频延迟调整（秒）
            let videoTimestamp = 0;
            let audioTimestamp = 0;
            let lastVideoFrameTime = 0;
            let videoFrameCount = 0;
            let fpsStartTime = Date.now();
            let socket = null;
            let reconnectAttempts = 0;
            const MAX_RECONNECT_ATTEMPTS = 10;
            
            // 音频播放队列
            const audioQueue = [];
            const MAX_AUDIO_QUEUE = 100;
            const MIN_AUDIO_QUEUE = 10;
            
            // 初始化Socket.IO连接
            function initSocket() {
                // 配置Socket.IO连接，增强自动重连机制
                socket = io({
                    reconnection: true,              // 启用自动重连
                    reconnectionAttempts: MAX_RECONNECT_ATTEMPTS, // 最大重连次数
                    reconnectionDelay: 1000,         // 重连延迟（毫秒）
                    reconnectionDelayMax: 5000,      // 最大重连延迟（毫秒）
                    timeout: 20000,                  // 连接超时（毫秒）
                    transports: ['websocket', 'polling'] // 优先使用websocket
                });
                
                // 连接事件处理
                socket.on('connect', () => {
                    console.log('连接到服务器');
                    connectionStatus.textContent = '已连接';
                    connectionStatus.style.color = '#4CAF50';
                    syncStatus.textContent = '连接成功，等待流数据...';
                    syncStatus.style.color = '#4CAF50';
                    reconnectAttempts = 0; // 重置重连计数
                });
                
                socket.on('disconnect', (reason) => {
                    console.log('与服务器断开连接，原因:', reason);
                    connectionStatus.textContent = '断开连接';
                    connectionStatus.style.color = '#ff0000';
                    syncStatus.textContent = `断开连接: ${reason}`;
                    syncStatus.style.color = '#ff9800';
                });
                
                socket.on('connect_error', (error) => {
                    console.error('连接错误:', error);
                    connectionStatus.textContent = '连接错误';
                    connectionStatus.style.color = '#f44336';
                    syncStatus.textContent = '连接错误，正在重试...';
                    syncStatus.style.color = '#f44336';
                });
                
                socket.on('reconnect_attempt', (attempt) => {
                    reconnectAttempts = attempt;
                    console.log(`尝试重连 (${attempt}/${MAX_RECONNECT_ATTEMPTS})`);
                    connectionStatus.textContent = `重连中... (${attempt}/${MAX_RECONNECT_ATTEMPTS})`;
                    connectionStatus.style.color = '#ff9800';
                });
                
                socket.on('reconnect', (attempt) => {
                    console.log('重连成功，尝试次数:', attempt);
                    connectionStatus.textContent = '已重连';
                    connectionStatus.style.color = '#4CAF50';
                    syncStatus.textContent = '重连成功，恢复流数据...';
                    syncStatus.style.color = '#4CAF50';
                });
                
                socket.on('reconnect_error', (error) => {
                    console.error('重连错误:', error);
                    connectionStatus.textContent = '重连失败';
                    connectionStatus.style.color = '#f44336';
                });
                
                socket.on('reconnect_failed', () => {
                    console.error('重连失败，已达到最大尝试次数');
                    connectionStatus.textContent = '重连失败';
                    connectionStatus.style.color = '#f44336';
                    syncStatus.textContent = '重连失败，已达到最大尝试次数';
                    syncStatus.style.color = '#f44336';
                });
            }
            
            // 初始化Socket.IO连接
            initSocket();
            
            // 初始化Socket.IO事件监听器
            function initSocketListeners() {
                // 处理视频帧
                socket.on('video_frame', (data) => {
                    if (!isPlaying) return;
                    
                    try {
                        const img = new Image();
                        img.onload = () => {
                            // 绘制视频帧
                            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                            
                            // 更新视频时间戳
                            videoTimestamp = data.timestamp;
                            
                            // 计算帧率
                            videoFrameCount++;
                            const now = Date.now();
                            if (now - fpsStartTime >= 1000) {
                                videoFps.textContent = videoFrameCount;
                                videoFrameCount = 0;
                                fpsStartTime = now;
                            }
                            
                            // 检查音画同步
                            checkSync();
                        };
                        
                        img.src = 'data:image/jpeg;base64,' + data.data;
                        
                    } catch (error) {
                        console.error('处理视频帧错误:', error);
                    }
                });
                
                // 处理音频帧
                socket.on('audio_frame', (data) => {
                    try {
                        // 更新音频时间戳
                        audioTimestamp = data.timestamp;
                        
                        // 解码base64音频数据
                        const audioData = Uint8Array.from(atob(data.data), c => c.charCodeAt(0));
                        
                        // 创建音频缓冲区
                        const buffer = audioContext.createBuffer(
                            data.channels,
                            Math.floor(audioData.length / (data.channels * 2)), // 假设16位音频
                            data.sample_rate
                        );
                        
                        // 填充音频数据
                        const view = new DataView(audioData.buffer);
                        let offset = 0;
                        for (let channel = 0; channel < data.channels; channel++) {
                            const channelData = buffer.getChannelData(channel);
                            for (let i = 0; i < channelData.length; i++) {
                                channelData[i] = view.getInt16(offset, true) / 32768;
                                offset += 2;
                            }
                        }
                        
                        // 添加到音频播放队列，无论是否正在播放
                        audioQueue.push({ buffer, timestamp: data.timestamp });
                        console.log(`收到音频帧，当前队列长度: ${audioQueue.length}`);
                        
                        // 智能队列管理：根据网络条件动态调整队列大小
                        const TARGET_QUEUE_SIZE = 30;
                        if (audioQueue.length > MAX_AUDIO_QUEUE) {
                            // 队列过长，移除旧数据
                            const excess = audioQueue.length - TARGET_QUEUE_SIZE;
                            audioQueue.splice(0, excess);
                            console.log(`队列过长，已移除 ${excess} 帧，当前队列长度: ${audioQueue.length}`);
                        }
                        
                    } catch (error) {
                        console.error('处理音频帧错误:', error);
                        // 在浏览器中不需要traceback
                    }
                });
                
                // 处理心跳包
                socket.on('heartbeat', (data) => {
                    console.log('收到心跳包:', data);
                    // 更新流状态
                    if (data.stream_alive) {
                        syncStatus.textContent = isPlaying ? '播放中...' : '流已连接，等待播放';
                        syncStatus.style.color = '#4CAF50';
                    } else {
                        syncStatus.textContent = '流数据异常';
                        syncStatus.style.color = '#ff9800';
                    }
                });
            }
            
            // 音频播放控制器
            let audioController = {
                isPlaying: false,
                nextPlayTime: 0,
                lastUpdateTime: 0,
                intervalId: null,
                
                start: function() {
                    if (this.isPlaying) return;
                    this.isPlaying = true;
                    this.nextPlayTime = audioContext.currentTime;
                    this.lastUpdateTime = Date.now();
                    
                    // 使用setInterval持续处理队列，确保即使队列为空也能继续监听
                    if (!this.intervalId) {
                        const that = this;
                        this.intervalId = setInterval(function() {
                            that._processQueue();
                        }, 50); // 每50ms检查一次队列
                    }
                    
                    console.log('音频控制器已启动');
                },
                
                stop: function() {
                    this.isPlaying = false;
                    if (this.intervalId) {
                        clearInterval(this.intervalId);
                        this.intervalId = null;
                    }
                    console.log('音频控制器已停止');
                },
                
                reset: function() {
                    this.nextPlayTime = audioContext.currentTime;
                    this.lastUpdateTime = Date.now();
                    audioQueue.length = 0;
                    console.log('音频控制器已重置');
                },
                
                _processQueue: function() {
                    if (!this.isPlaying || !audioContext) {
                        return;
                    }
                    
                    try {
                        const currentTime = audioContext.currentTime;
                        
                        // 如果队列为空，返回但继续下一次检查
                        if (audioQueue.length === 0) {
                            return;
                        }
                        
                        // 计算队列中可播放的帧数
                        let framesToPlay = 0;
                        let totalDuration = 0;
                        
                        // 确保播放时间不会落后太多
                        if (this.nextPlayTime < currentTime) {
                            this.nextPlayTime = currentTime;
                        }
                        
                        // 计算能在当前时间窗口内播放的帧数
                        for (let i = 0; i < audioQueue.length; i++) {
                            const frame = audioQueue[i];
                            totalDuration += frame.buffer.duration;
                            if (this.nextPlayTime + totalDuration <= currentTime + 0.5) { // 预播放0.5秒的音频
                                framesToPlay++;
                            } else {
                                break;
                            }
                        }
                        
                        // 至少播放1帧，确保音频持续播放
                        framesToPlay = Math.max(1, framesToPlay);
                        
                        console.log(`处理音频队列，当前队列长度: ${audioQueue.length}，计划播放帧数: ${framesToPlay}`);
                        
                        // 播放音频帧
                        for (let i = 0; i < framesToPlay; i++) {
                            if (audioQueue.length === 0) break;
                            
                            const audioFrame = audioQueue.shift();
                            const source = audioContext.createBufferSource();
                            source.buffer = audioFrame.buffer;
                            source.connect(audioContext.destination);
                            
                            // 精确计算播放时间
                            const bufferDuration = audioFrame.buffer.duration;
                            
                            try {
                                source.start(this.nextPlayTime);
                                this.nextPlayTime += bufferDuration;
                                console.log(`播放音频帧，时长: ${bufferDuration.toFixed(3)}s，下次播放时间: ${this.nextPlayTime.toFixed(3)}s`);
                            } catch (error) {
                                console.error('播放音频片段错误:', error);
                                // 调整下一次播放时间，跳过当前片段
                                this.nextPlayTime = currentTime;
                                continue;
                            }
                        }
                        
                    } catch (error) {
                        console.error('播放音频队列错误:', error);
                        // 错误处理：重置播放状态
                        this.nextPlayTime = audioContext.currentTime;
                    }
                }
            };
            
            // 检查音画同步
            function checkSync() {
                if (videoTimestamp > 0 && audioTimestamp > 0) {
                    // 计算音画延迟
                    const delay = Math.abs(videoTimestamp - audioTimestamp) * 1000; // 转换为毫秒
                    delayDisplay.textContent = delay.toFixed(2);
                    
                    // 更新同步状态
                    if (delay < 100) {
                        syncStatus.textContent = '音画同步良好';
                        syncStatus.style.color = '#4CAF50';
                    } else if (delay < 300) {
                        syncStatus.textContent = '音画基本同步';
                        syncStatus.style.color = '#ff9800';
                    } else {
                        syncStatus.textContent = '音画不同步';
                        syncStatus.style.color = '#f44336';
                    }
                } else if (videoTimestamp > 0) {
                    syncStatus.textContent = '视频流正常';
                    syncStatus.style.color = '#4CAF50';
                }
            }
            
            // 初始化音频上下文
            function initAudioContext() {
                if (!audioContext) {
                    try {
                        audioContext = new (window.AudioContext || window.webkitAudioContext)();
                        console.log('音频上下文初始化成功');
                    } catch (error) {
                        console.error('音频上下文初始化失败:', error);
                        alert('无法初始化音频上下文，请检查浏览器权限设置');
                        return false;
                    }
                }
                return true;
            }
            
            // 播放/暂停控制
            function togglePlay() {
                if (isPlaying) {
                    // 暂停
                    isPlaying = false;
                    audioController.stop();
                    playBtn.textContent = '播放';
                    syncStatus.textContent = '已暂停';
                    syncStatus.style.color = '#ff9800';
                    // 清空音频队列
                    audioQueue.length = 0;
                } else {
                    // 播放
                    initAudioContext();
                    isPlaying = true;
                    playBtn.textContent = '暂停';
                    syncStatus.textContent = '播放中...';
                    syncStatus.style.color = '#4CAF50';
                    // 启动音频控制器
                    audioController.start();
                }
            }
            
            // 重置流
            function syncAudioVideo() {
                // 重置时间戳和队列
                videoTimestamp = 0;
                audioTimestamp = 0;
                // 重置音频控制器
                audioController.reset();
                
                // 重新初始化Socket连接
                socket.disconnect();
                setTimeout(() => {
                    initSocket();
                    initSocketListeners();
                    syncStatus.textContent = '重新连接中...';
                    syncStatus.style.color = '#ff9800';
                }, 1000);
                
                console.log('流已重置');
            }
            
            // 初始化
            window.onload = () => {
                console.log('页面加载完成');
                // 初始化Socket事件监听器
                initSocketListeners();
            };
        </script>
    </body>
    </html>
    '''
    return render_template_string(html)

# Socket.IO事件
@socketio.on('connect')
def handle_connect():
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Socket.IO 客户端已连接，SID: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Socket.IO 客户端已断开连接，SID: {request.sid}")

@socketio.on('connect_error')
def handle_connect_error(e):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Socket.IO 连接错误: {e}")
    traceback.print_exc()

if __name__ == '__main__':
    print('启动音画同步直播服务器...')
    print(f'RTMP流地址: {rtmp_url}')
    print('Web服务器地址: http://0.0.0.0:5000')
    
    # 启动流处理器
    stream_processor.start()
    
    # 启动Flask服务器
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)

