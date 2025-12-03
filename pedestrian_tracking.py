import cv2
import numpy as np
import os
import time
from ultralytics import YOLO

# 确保output文件夹存在
os.makedirs('output', exist_ok=True)

# 加载YOLOv8模型并尝试使用GPU加速
device = 0  # 0表示第一个GPU，如果有的话
person_model = YOLO('yolov8n.pt')  # 行人检测模型

# 检查是否有可用的GPU
try:
    import torch
    if torch.cuda.is_available():
        print(f"检测到可用的GPU: {torch.cuda.get_device_name(0)}")
        print("将使用GPU进行加速")
    else:
        print("未检测到可用的GPU，将使用CPU")
        device = 'cpu'
except ImportError:
    print("PyTorch未安装，无法检查GPU可用性")
    device = 'cpu'

# 读取指定的视频文件
# video_path = 'static/test_video/1.mp4'
video_path = 'rtmp://192.168.0.220:1935/live/stream_name'

cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print(f"无法打开视频文件: {video_path}")
    exit()

# 获取视频基本信息
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"视频信息: {width}x{height} @ {fps}fps, 总帧数: {total_frames}")
print(f"开始处理视频: {video_path}")
print("每5帧保存一次预测结果到output文件夹")

frame_count = 0
start_time = time.time()  # 开始时间用于计算平均FPS
fps_list = []  # 存储每帧的FPS以便计算平均FPS
person_count_list = []  # 存储每帧检测到的行人数量

# 主循环
while True:
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        print("视频处理完成")
        break
    
    # 调整帧大小以提高处理速度
    frame = cv2.resize(frame, (640, 480))
    
    # 记录当前帧的开始时间
    frame_start_time = time.time()
    
    # 使用YOLOv8进行全类别检测，但后续会对不同类别使用不同颜色的框
    person_results = person_model.track(
        frame, 
        persist=True, 
        # 不指定classes，检测所有类别
        classes=[0],  # 仅检测人类（class 0）
        verbose=False, 
        device=device,
        conf=0.3,  # 置信度阈值
        iou=0.45   # IOU阈值
    )

    # 自定义绘图函数：为人类添加高斯模糊马赛克效果，其他物体使用淡灰色框
    def custom_plot(results, frame):
        # 创建帧的副本
        annotated_frame = frame.copy()
        
        # 获取所有检测框
        boxes = results.boxes
        
        if boxes is not None:
            # 获取类别ID、置信度、跟踪ID和边界框坐标
            class_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
            confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else []
            track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
            xyxy = boxes.xyxy.cpu().numpy()
            
            # 获取类别名称
            names = results.names
            
            # 绘制每个检测框
            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                class_id = class_ids[i] if i < len(class_ids) else 0
                conf = confidences[i] if i < len(confidences) else 0
                
                # 确定框的颜色：人（class 0）使用原样式颜色，其他使用淡灰色
                if class_id == 0:  # 人类
                    # 使用YOLO默认的颜色（这里使用蓝色作为示例）
                    color = (255, 0, 0)  # BGR格式
                    thickness = 1
                    
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
                else:  # 其他物体
                    # 使用淡灰色
                    color = (192, 192, 192)  # BGR格式的淡灰色
                    thickness = 1
                
                # 绘制矩形框
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                
        
        return annotated_frame
    
    # 使用自定义绘图函数绘制检测结果
    annotated_frame = custom_plot(person_results[0], frame)
    
    # 获取检测到的行人数量
    person_count = len(person_results[0].boxes) if person_results[0].boxes is not None else 0
    
    person_count_list.append(person_count)
    
    # 计算当前帧的FPS
    frame_time = time.time() - frame_start_time
    current_fps = 1.0 / frame_time if frame_time > 0 else 0
    fps_list.append(current_fps)
    
    # 计算平均FPS
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    avg_person_count = sum(person_count_list) / len(person_count_list) if person_count_list else 0
    
    # 在帧上显示统计信息
    cv2.putText(annotated_frame, f'Current FPS: {current_fps:.1f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f'Avg FPS: {avg_fps:.1f}', (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f'Persons: {person_count}', (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f'Frame: {frame_count}/{total_frames}', (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    print(f"当前帧的行人数量: {person_count},avg_fps:{avg_fps:.1f}")
    frame_count += 1

# 释放资源
cap.release()
cv2.destroyAllWindows()
