import av
import cv2
import numpy as np
import base64
import time
import threading
from flask import Flask, render_template_string, render_template, request
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
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
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
    return render_template('index.html')

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

