import av
import cv2
import numpy as np
import traceback


rtmp_url = 'rtmp://192.168.1.2:1935/live/123'


print("PyAV版本:", av.__version__)
print("OpenCV版本:", cv2.__version__)


# 打开RTMP流并配置
try:
    print(f"尝试打开RTMP流: {rtmp_url}")
    # 使用测试脚本中成功的方法打开流
    container = av.open(rtmp_url, mode='r')
    print("容器打开成功")
    
    # 打印流信息
    print("流信息:")
    video_stream = None
    audio_stream = None
    for stream in container.streams:
        print(f"- {stream.type}: index={stream.index}, codec={stream.codec_context.name}")
        if stream.type == 'video':
            video_stream = stream
        elif stream.type == 'audio':
            audio_stream = stream


    print("流已打开，正在处理...")
    
    # 仅处理前几帧以测试
    frame_count = 0
    max_frames = 9999999999
    
    # 遍历流中的数据包
    for packet in container.demux():
        if frame_count >= max_frames:
            break
            
        try:
            # 如果是视频帧
            if packet.stream.type == 'video':
                print(f"处理视频数据包: stream={packet.stream.index}, size={packet.size}")
                for frame in packet.decode():
                    frame_count += 1
                    print(f"  解码视频帧 #{frame_count}: width={frame.width}, height={frame.height}")
                    
                    try:
                        # 将 PyAV 帧转换为 numpy 数组 (OpenCV 格式)
                        img = frame.to_ndarray(format='bgr24')
                        print(f"  转换为numpy数组: shape={img.shape}")
                        
                        # 调整图像大小以便显示
                        img_small = cv2.resize(img, (640, 480))
                        
                        # 显示视频
                        cv2.imshow('Video', img_small)
                        key = cv2.waitKey(1) 
                        if key & 0xFF == ord('q'):
                            frame_count = max_frames
                            break
                    except Exception as e:
                        print(f"  处理视频帧时出错: {e}")
                        traceback.print_exc()


            # 如果是音频帧
            elif packet.stream.type == 'audio':
                print(f"处理音频数据包: stream={packet.stream.index}, size={packet.size}")
                for frame in packet.decode():
                    try:
                        # 获取音频数据 (numpy array)
                        audio_data = frame.to_ndarray()
                        # 在这里处理音频，例如发送给语音识别API
                        # 注意：audio_data 的形状通常是 (samples, channels)
                        print(f"  收到音频帧: shape={audio_data.shape}")
                    except Exception as e:
                        print(f"  处理音频帧时出错: {e}")
                        traceback.print_exc()
        except Exception as e:
            print(f"处理数据包时出错: {e}")
            traceback.print_exc()


except KeyboardInterrupt:
    print("停止处理")
except Exception as e:
    print(f"发生错误: {type(e).__name__}: {e}")
    traceback.print_exc()
finally:
    print("清理资源...")
    cv2.destroyAllWindows()
    try:
        container.close()
        print("容器已关闭")
    except:
        pass