#!/usr/bin/env python3
"""
立体 SLAM Web 可视化测试
在 9704 端口提供 Web 界面，显示实时摄像头视频和点云
"""

import cv2
import numpy as np
import time
import threading
from flask import Flask, Response, jsonify
import logging

from src.stereo_slam import StereoSLAM

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask 应用
app = Flask(__name__)

# 全局变量存储最新帧和点云数据
latest_frame = None
latest_left_frame = None
latest_right_frame = None
point_cloud_data = None
frame_stats = {}
lock = threading.Lock()

# 摄像头和 SLAM 全局变量
cap = None
slam = None


def create_synthetic_stereo_images(width=2560, height=720):
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


def project_3d_to_2d(points_3d, camera_pose, K, image_shape):
    """将 3D 点投影到 2D 图像平面"""
    if len(points_3d) == 0:
        return []
    
    camera_pose_inv = np.linalg.inv(camera_pose)
    points_cam = (camera_pose_inv[:3, :3] @ points_3d.T + camera_pose_inv[:3, 3:4]).T
    
    valid_mask = points_cam[:, 2] > 0.1
    if not np.any(valid_mask):
        return []
    
    points_cam_valid = points_cam[valid_mask]
    points_2d = (K @ points_cam_valid.T).T
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]
    
    h, w = image_shape[:2]
    in_image = (
        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) &
        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
    )
    
    return points_2d[in_image]


def process_frame_for_display(left_img, right_img, slam_instance):
    """处理帧并返回带点云显示的图像"""
    # 处理帧
    result = slam_instance.process_frame(left_img, right_img)
    
    # 获取 3D 点并投影
    positions = slam_instance.map.get_3d_points_array()
    projected_points = []
    
    if len(positions) > 0:
        camera_pose = slam_instance.get_camera_pose()
        K = slam_instance.K.copy()
        projected = project_3d_to_2d(positions, camera_pose, K, left_img.shape)
        projected_points = [(int(p[0]), int(p[1]), p[2] if len(p) > 2 else 1.0) for p in projected]
    
    # 创建显示图像 - 左右并排
    display_img = np.hstack((left_img, right_img))
    
    # 在左图上绘制投影点
    for pt in projected_points:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < left_img.shape[1] and 0 <= y < left_img.shape[0]:
            depth = pt[2]
            if depth < 3:
                color = (0, 255, 0)
            elif depth < 6:
                color = (0, 255, 255)
            else:
                color = (0, 100, 255)
            cv2.circle(display_img, (x, y), 3, color, -1)
    
    # 添加文本信息
    cam_pos = slam_instance.get_camera_position()
    
    cv2.putText(display_img, f"3D Points: {len(positions)}", (20, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_img, f"Cam: ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f})",
               (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # 添加标签
    mid_w = left_img.shape[1]
    cv2.putText(display_img, "LEFT (with 3D points)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(display_img, "RIGHT", (mid_w + 20, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return display_img, len(positions), cam_pos


def generate_frame():
    """生成视频帧流 - 使用正确的 MJPEG 格式"""
    global cap, slam, latest_frame, latest_left_frame, latest_right_frame, point_cloud_data, frame_stats
    
    # 初始化 SLAM
    if slam is None:
        logger.info("初始化 Stereo SLAM...")
        slam = StereoSLAM(
            device_id=0,
            baseline=0.065,
            image_width=2560,
            image_height=720,
            fov_horizontal=100.0,
            debug_mode=False
        )
        logger.info("SLAM 初始化完成")
    
    # 打开摄像头
    if cap is None:
        logger.info("尝试打开摄像头...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            logger.warning("无法打开摄像头，将使用合成图像")
    
    frame_count = 0
    
    while True:
        # 获取图像
        use_synthetic = False
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                use_synthetic = True
            else:
                h, w = frame.shape[:2]
                mid = w // 2
                left_img = frame[:, :mid, :]
                right_img = frame[:, mid:, :]
                
                if h != 720 or mid != 1280:
                    use_synthetic = True
        else:
            use_synthetic = True
        
        if use_synthetic:
            left_img, right_img = create_synthetic_stereo_images()
        
        # 处理帧并获取显示图像
        display_img, num_points, cam_pos = process_frame_for_display(left_img, right_img, slam)
        frame_count += 1
        
        # 更新全局变量
        with lock:
            latest_frame = display_img.copy()
            latest_left_frame = left_img.copy()
            latest_right_frame = right_img.copy()
            frame_stats = {
                'frame_count': frame_count,
                'num_points': num_points,
                'camera_pos': cam_pos.tolist(),
                'elapsed': time.time()
            }
        
        # 编码为 JPEG
        success, buffer = cv2.imencode('.jpg', display_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            time.sleep(0.033)
            continue
        
        frame_bytes = buffer.tobytes()
        
        # 输出 MJPEG 格式
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


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
                <img src="{{ url_for('video_feed') }}" alt="Video Stream" width="640" height="360">
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
                    <span class="stat-label">Camera Position:</span>
                    <span class="stat-value" id="campos">--</span>
                </div>
            </div>
        </div>
        
        <script>
            function updateStats() {
                fetch('/stats')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('frame').textContent = data.frame_count;
                        document.getElementById('points').textContent = data.num_points;
                        document.getElementById('campos').textContent = 
                            '(' + data.camera_pos[0].toFixed(2) + ', ' + 
                            data.camera_pos[1].toFixed(2) + ', ' + 
                            data.camera_pos[2].toFixed(2) + ')';
                    })
                    .catch(err => console.error('Error:', err));
            }
            
            setInterval(updateStats, 500);
            updateStats();
        </script>
    </body>
    </html>
    '''
    return html


@app.route('/video_feed')
def video_feed():
    """视频流路由 - 返回 MJPEG 流"""
    return Response(generate_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def get_stats():
    """获取统计数据"""
    with lock:
        return jsonify(frame_stats)


def run_web_server(port=9704):
    """运行 Web 服务器"""
    logger.info(f"启动 Web 服务器在端口 {port}...")
    logger.info(f"打开浏览器访问：http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, threaded=True)


if __name__ == "__main__":
    print("=" * 60)
    print("Stereo SLAM Web Visualization")
    print("=" * 60)
    print("打开浏览器访问：http://localhost:9704")
    print("按 Ctrl+C 退出")
    print("=" * 60)
    
    run_web_server(9704)