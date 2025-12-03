import time
import cv2
import numpy as np
from flask import Flask, Response
from ultralytics import YOLO
import torch

# 创建Flask应用
app = Flask(__name__)

# RTMP流地址
rtmp_url = 'rtmp://192.168.1.2:1935/live/stream_name'

# 加载YOLOv8模型并尝试使用GPU加速
device = 0  # 0表示第一个GPU，如果有的话
person_model = YOLO('yolov8n.pt')  # 行人检测模型

# 检查是否有可用的GPU
try:
    if torch.cuda.is_available():
        print(f"检测到可用的GPU: {torch.cuda.get_device_name(0)}")
        print("将使用GPU进行加速")
    else:
        print("未检测到可用的GPU，将使用CPU")
        device = 'cpu'
except ImportError:
    print("PyTorch未安装，无法检查GPU可用性")
    device = 'cpu'

# 初始化视频捕获
cap = cv2.VideoCapture(rtmp_url)

# 检查视频是否成功打开
if not cap.isOpened():
    print(f"无法打开RTMP流: {rtmp_url}")
    exit()

# 获取视频基本信息
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"视频信息: {width}x{height} @ {fps}fps")
print(f"开始处理RTMP流: {rtmp_url}")

# 自定义绘图函数：为人类添加高斯模糊马赛克效果
def apply_pedestrian_mosaic(frame, results):
    # 创建帧的副本
    annotated_frame = frame.copy()
    
    # 获取所有检测框
    boxes = results.boxes
    
    if boxes is not None:
        # 获取边界框坐标
        xyxy = boxes.xyxy.cpu().numpy()
        
        # 绘制每个检测框
        for (x1, y1, x2, y2) in xyxy:
            # 确保坐标在有效范围内
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(annotated_frame.shape[1] - 1, int(x2))
            y2 = min(annotated_frame.shape[0] - 1, int(y2))
            
            # 确保框至少有一定大小
            if (x2 - x1 > 10) and (y2 - y1 > 10):
                # 提取行人框区域
                person_roi = annotated_frame[y1:y2, x1:x2]
                
                # 应用高斯模糊（参数5表示模糊程度，可以根据需要调整）
                blurred_roi = cv2.GaussianBlur(person_roi, (7, 7), 5)
                
                # 将模糊后的区域放回原图
                annotated_frame[y1:y2, x1:x2] = blurred_roi
                
                # 绘制矩形框
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
    
    return annotated_frame

# 生成视频流的函数
def generate_frames():
    frame_count = 0
    start_time = time.time()
    fps_list = []
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("RTMP流读取失败，尝试重新连接...")
            # 尝试重新连接
            cap.release()
            cap.open(rtmp_url)
            time.sleep(1)
            continue
        
        # 调整帧大小以提高处理速度
        frame = cv2.resize(frame, (640, 480))
        
        # 记录当前帧的开始时间
        frame_start_time = time.time()
        
        # 使用YOLOv8进行行人检测
        person_results = person_model.track(
            frame, 
            persist=True, 
            classes=[0],  # 仅检测人类（class 0）
            verbose=False, 
            device=device,
            conf=0.3,  # 置信度阈值
            iou=0.45   # IOU阈值
        )
        
        # 应用行人马赛克效果
        annotated_frame = apply_pedestrian_mosaic(frame, person_results[0])
        
        # 获取检测到的行人数量
        person_count = len(person_results[0].boxes) if person_results[0].boxes is not None else 0
        
        # 计算当前帧的FPS
        frame_time = time.time() - frame_start_time
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        fps_list.append(current_fps)
        if len(fps_list) > 10:  # 只保留最近10帧的FPS用于计算平均
            fps_list.pop(0)
        avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
        
        # 在帧上显示统计信息
        cv2.putText(annotated_frame, f'Current FPS: {current_fps:.1f}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Avg FPS: {avg_fps:.1f}', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Persons: {person_count}', (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f'Frame: {frame_count}', (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        frame_count += 1
        
        # 将帧转换为JPEG格式
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        
        # 将JPEG数据转换为字节流
        frame_bytes = buffer.tobytes()
        
        # 使用multipart/x-mixed-replace格式生成视频流
        yield (b'--frame\r\n' 
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# 定义视频流路由
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 定义主页路由
@app.route('/')
def index():
    return '''
    <html>
    <head>
        <title>RTMP Live Stream with Pedestrian Mosaic</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                background-color: #f0f0f0;
                margin: 0;
                padding: 0;
            }
            h1 {
                color: #333;
                margin-top: 20px;
            }
            .stream-container {
                margin: 20px auto;
                max-width: 800px;
                background-color: #fff;
                padding: 10px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            img {
                width: 100%;
                height: auto;
                border-radius: 4px;
            }
            .info {
                margin-top: 10px;
                color: #666;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <h1>RTMP Live Stream with Pedestrian Mosaic</h1>
        <div class="stream-container">
            <img src="/video_feed" alt="Live Stream">
            <div class="info">
                <p>实时直播流，已应用行人马赛克处理</p>
            </div>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    # 启动Flask应用，使用0.0.0.0允许外部访问
    app.run(host='0.0.0.0', port=5000, debug=False)
