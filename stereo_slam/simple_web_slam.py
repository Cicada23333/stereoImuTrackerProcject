#!/usr/bin/env python3
"""
简单的立体 SLAM Web 可视化
使用线程共享帧，而不是生成器
"""

import cv2
import numpy as np
import time
import threading
from flask import Flask, Response, jsonify
import logging

from src.core.stereo_slam import StereoSLAM
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 共享帧缓冲区
shared_frame = None
latest_result = None
processed_frame_count = 0
frame_lock = threading.Lock()
running = True

# 摄像头和 SLAM
cap = None
slam = None


def create_synthetic_stereo_images(width=1280, height=720):
    """创建合成的立体图像对"""
    left_img = np.zeros((height, width, 3), dtype=np.uint8)
    np.random.seed(42)
    
    for _ in range(500):
        x = np.random.randint(50, width - 50)
        y = np.random.randint(50, height - 50)
        radius = np.random.randint(3, 10)
        color = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
        cv2.circle(left_img, (x, y), radius, color, -1)
    
    right_img = left_img.copy()
    right_img[:, :-30] = left_img[:, 30:]
    right_img[:, -30:] = 0
    
    return left_img, right_img


def create_synthetic_stereo_frame(width=2560, height=720, eye_order="left-right"):
    """创建一张左右拼接的合成双目图。"""
    eye_width = width // 2
    left_img, right_img = create_synthetic_stereo_images(eye_width, height)
    if eye_order == "right-left":
        return np.hstack((right_img, left_img))
    return np.hstack((left_img, right_img))


def normalize_camera_frame(frame, expected_width=2560, expected_height=720):
    """将摄像头实际输出裁剪成库期望的 2560x720 拼接图。"""
    height, width = frame.shape[:2]
    if width != expected_width:
        return None
    if height == expected_height:
        return frame
    if height > expected_height:
        top = (height - expected_height) // 2
        return frame[top:top + expected_height, :].copy()
    return None


def frame_generator():
    """后台线程：持续捕获和处理帧"""
    global cap, slam, shared_frame, latest_result, processed_frame_count, running
    
    # 初始化 SLAM
    logger.info("初始化 Stereo SLAM...")
    slam = StereoSLAM(
        device_id=0,
        baseline=0.065,
        image_width=1280,
        image_height=720,
        stereo_width=2560,
        fov_horizontal=100.0,
        eye_order="right-left",
        debug_mode=False
    )
    logger.info("SLAM 初始化完成")
    
    # 打开摄像头 - 尝试多种分辨率
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)   # Windows推荐
        
    # 尝试多种分辨率，从常见到高分辨率
    resolutions = [
        (2560, 720),   # 1280x720 左眼 + 1280x720 右眼
    ]
    
    use_synthetic = False
    actual_width, actual_height = 640, 480  # 默认值
    
    for width, height in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # 读取一帧来验证
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            actual_width, actual_height = test_frame.shape[1], test_frame.shape[0]
            logger.info(f"摄像头成功打开，实际分辨率：{actual_width}x{actual_height}")
            break
    else:
        # 所有分辨率都失败
        logger.warning("无法打开摄像头，将使用合成图像")
        use_synthetic = True
        cap.release()
        cap = None
    
    frame_count = 0
    
    while running:
        use_synthetic_frame = use_synthetic
        stereo_frame = None
        
        # 获取图像
        if not use_synthetic:
            ret, frame = cap.read()
            if not ret or frame is None:
                use_synthetic_frame = True
            else:
                h, w = frame.shape[:2]
                stereo_frame = normalize_camera_frame(frame)
                if stereo_frame is None:
                    use_synthetic_frame = True
        
        if use_synthetic_frame:
            stereo_frame = create_synthetic_stereo_frame(eye_order=slam.eye_order)

        left_img, right_img = slam.split_stereo_image(stereo_frame)
        
        # 处理帧
        result = slam.process_stereo_image(stereo_frame)
        frame_count += 1
        
        # 获取地图点统计和当前帧真实观测点
        positions = slam.map.get_3d_points_array()
        current_observations = slam.get_last_observations()
        
        # 创建显示图像 - 左右并排
        display_img = np.hstack((left_img.copy(), right_img.copy()))
        
        mid_w = left_img.shape[1]  # 左图的宽度
        
        # 当前帧观测直接画在 ORB 匹配特征位置上；这比投影整张历史地图更适合调试“是否绑定到特征”。
        for obs in current_observations:
            left_x, left_y = int(obs["left"][0]), int(obs["left"][1])
            right_x, right_y = int(obs["right"][0]), int(obs["right"][1])
            depth = float(obs["depth"])

            if depth < 3:
                left_color = (0, 255, 0)
                right_color = (255, 0, 0)
            elif depth < 6:
                left_color = (0, 255, 255)
                right_color = (255, 255, 0)
            else:
                left_color = (0, 120, 255)
                right_color = (120, 120, 255)

            if 0 <= left_x < left_img.shape[1] and 0 <= left_y < left_img.shape[0]:
                cv2.circle(display_img, (left_x, left_y), 3, left_color, -1)
                cv2.circle(display_img, (left_x, left_y), 6, left_color, 1)

            if 0 <= right_x < right_img.shape[1] and 0 <= right_y < right_img.shape[0]:
                cv2.circle(display_img, (right_x + mid_w, right_y), 3, right_color, -1)
                cv2.circle(display_img, (right_x + mid_w, right_y), 6, right_color, 1)
        
        # 添加文本信息
        cam_pos = slam.get_camera_position()
        
        cv2.putText(display_img, f"3D Points: {len(positions)}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_img, f"Observed: {len(current_observations)}", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(display_img, f"Cam: ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f})",
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.putText(display_img, "LEFT", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_img, "RIGHT", (mid_w + 20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 更新共享帧
        with frame_lock:
            shared_frame = display_img.copy()
            latest_result = result
            processed_frame_count = frame_count
        
        time.sleep(0.033)  # 约 30 FPS


def get_frame():
    """获取当前帧并编码为 JPEG"""
    with frame_lock:
        if shared_frame is not None:
            success, buffer = cv2.imencode('.jpg', shared_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                return buffer.tobytes()
    return None


@app.route('/')
def index():
    """主页面"""
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stereo SLAM Web Visualization</title>
        <style>
            body {
                margin: 0;
                padding: 20px;
                background-color: #1a1a2e;
                color: #eee;
                font-family: Arial, sans-serif;
            }
            h1 {
                text-align: center;
                color: #00ff88;
            }
            .container {
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .video-box {
                text-align: center;
                margin-bottom: 20px;
            }
            .video-box h3 {
                margin: 5px 0;
                color: #00ff88;
            }
            img {
                border: 2px solid #333;
                border-radius: 8px;
                max-width: 100%;
            }
            .stats {
                background-color: #16213e;
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
                min-width: 300px;
            }
            .stats h3 {
                margin-top: 0;
                color: #00ff88;
            }
            .stat-item {
                display: flex;
                justify-content: space-between;
                padding: 5px 0;
                border-bottom: 1px solid #333;
            }
            .stat-label {
                color: #aaa;
            }
            .stat-value {
                color: #00ff88;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <h1>Stereo SLAM Real-time Visualization</h1>
        
        <div class="container">
            <div class="video-box">
                <h3>Camera (with 3D Points)</h3>
                <img id="video" src="" alt="Video Stream" width="640" height="360">
            </div>
            
            <div class="stats">
                <h3>Statistics</h3>
                <div class="stat-item">
                    <span class="stat-label">Frame Count:</span>
                    <span class="stat-value" id="frame">--</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">3D Points:</span>
                    <span class="stat-value" id="points">--</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Current Observations:</span>
                    <span class="stat-value" id="observations">--</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Camera Position:</span>
                    <span class="stat-value" id="campos">--</span>
                </div>
            </div>
        </div>
        
        <script>
            // 使用 setInterval 刷新图片
            function updateFrame() {
                document.getElementById('video').src = '/frame.jpg?t=' + Date.now();
            }
            
            function updateStats() {
                fetch('/stats')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('frame').textContent = data.frame_count;
                        document.getElementById('points').textContent = data.num_points;
                        document.getElementById('observations').textContent = data.num_current_observations || 0;
                        document.getElementById('campos').textContent = 
                            '(' + data.camera_pos[0].toFixed(2) + ', ' + 
                            data.camera_pos[1].toFixed(2) + ', ' + 
                            data.camera_pos[2].toFixed(2) + ')';
                    })
                    .catch(err => console.error('Error:', err));
            }
            
            // 每 50ms 刷新图片
            setInterval(updateFrame, 50);
            // 每 500ms 刷新统计
            setInterval(updateStats, 500);
            updateStats();
        </script>
    </body>
    </html>
    '''
    return html


@app.route('/frame.jpg')
def frame():
    """返回当前帧的 JPEG 图像"""
    frame_data = get_frame()
    if frame_data:
        return Response(frame_data, mimetype='image/jpeg')
    return Response(b'', mimetype='image/jpeg')


@app.route('/stats')
def get_stats():
    """获取统计数据"""
    with frame_lock:
        stats = {
            'frame_count': 0,
            'num_points': 0,
            'camera_pos': [0.0, 0.0, 0.0]
        }
        if slam is not None:
            positions = slam.map.get_3d_points_array()
            cam_pos = slam.get_camera_position()
            stats = {
                'frame_count': processed_frame_count,
                'num_points': len(positions),
                'num_current_observations': (
                    latest_result.get('num_current_observations', 0)
                    if latest_result else 0
                ),
                'camera_pos': cam_pos.tolist(),
                'last_result': latest_result
            }
        return jsonify(stats)


def cleanup():
    """清理资源"""
    global running
    running = False
    if cap:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=" * 60)
    print("Stereo SLAM Web Visualization (Simple Version)")
    print("=" * 60)
    print("打开浏览器访问：http://localhost:9704")
    print("按 Ctrl+C 退出")
    print("=" * 60)
    
    # 启动后台帧生成线程
    thread = threading.Thread(target=frame_generator, daemon=True)
    thread.start()
    
    try:
        app.run(host='0.0.0.0', port=9704, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
