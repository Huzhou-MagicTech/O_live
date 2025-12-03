import time
import cv2
import numpy as np
from ffpyplayer.player import MediaPlayer

rtmp_url = 'rtmp://192.168.1.2:1935/live/stream_name'

def play_low_latency_stream(url):
    # --- 关键修改点：设置 FFmpeg 参数以降低延迟 ---
    ff_opts = {
        # 1. 'nobuffer': 最关键参数，告诉 FFmpeg 不要缓存数据，来多少解多少
        'fflags': 'nobuffer',
        
        # 2. 'low_delay': 告诉解码器这是一个低延迟流
        'flags': 'low_delay',
        
        # 3. 'framedrop': 如果 CPU 处理不过来，直接丢弃视频帧，优先保证音频和实时性
        'framedrop': True,
        
        # 4. 'strict': 允许使用实验性的参数（有时对低延迟有帮助）
        'strict': 'experimental',
        
        # 5. 减少分析流信息的时间（加快首帧显示）
        'probesize': '32',
        'analyzeduration': '0',
        
        # 6. RTMP 专用参数：指明是直播流
        'rtmp_live': 'live',
    }

    # 将参数传入 MediaPlayer
    player = MediaPlayer(url, ff_opts=ff_opts)
    
    print("正在以低延迟模式连接...")

    while True:
        frame, val = player.get_frame()
        
        if val == 'eof':
            print("流结束")
            break
        elif frame is None:
            # 没有帧时，等待时间极短，避免阻塞
            time.sleep(0.001)
            continue
        else:
            img, t = frame
            w, h = img.get_size()
            data = img.to_bytearray()[0]
            
            np_image = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
            cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
            
            # --- 同步逻辑优化 ---
            # 在低延迟模式下，我们希望尽可能快地播放
            # 只有当 val (需要等待的时间) 大于一定阈值时才 sleep
            # 这里的逻辑是：如果 FFmpeg 告诉我们“这一帧来早了”，我们稍微等一下
            # 但不要等太久，以免积压
            if val > 0:
                time.sleep(val)
        
        # 将 waitKey 设为 1ms，这是 OpenCV 窗口刷新的最小间隔
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    player.close_player()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play_low_latency_stream(rtmp_url)