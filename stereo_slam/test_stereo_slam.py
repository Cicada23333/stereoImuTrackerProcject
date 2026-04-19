#!/usr/bin/env python3
"""
立体 SLAM 测试脚本 - 带实时摄像头显示和点云可视化
参考 test/stereo_camera_slam.py 的实现方式
"""

import cv2
import numpy as np
import logging
import sys
import time

sys.path.insert(0, '.')

from src.stereo_slam import StereoSLAM


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("test_stereo_slam")


def create_synthetic_stereo_images(width=2560, height=800):
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


def run_stereo_slam_test(device_id=0, duration_seconds=5, use_synthetic=False):
    """运行立体 SLAM 测试 - 在摄像头视频上显示点云"""
    logger = setup_logging()
    
    print("=" * 60)
    print("Stereo SLAM - 实时摄像头视频 + 点云可视化")
    print("=" * 60)
    print(f"Device ID: {device_id}")
    print(f"Duration: {duration_seconds} seconds")
    print("=" * 60)
    
    # 初始化 SLAM
    slam = StereoSLAM(
        device_id=device_id,
        baseline=0.065,
        image_width=2560,
        image_height=800,
        fov_horizontal=100.0,
        debug_mode=True
    )
    
    # 打开摄像头
    cap = None
    if not use_synthetic:
        print(f"\n[INFO] 尝试打开摄像头 (索引：{device_id})...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
        
        if not cap.isOpened():
            print("[WARNING] 无法打开摄像头，使用合成图像")
            use_synthetic = True
        else:
            print("[INFO] 摄像头打开成功")
    
    frame_count = 0
    start_time = time.time()
    
    print(f"\n[INFO] 开始处理... 按 'q' 退出")
    
    try:
        while time.time() - start_time < duration_seconds:
            # 获取图像
            if use_synthetic:
                left_img, right_img = create_synthetic_stereo_images()
            else:
                ret, frame = cap.read()
                if not ret:
                    use_synthetic = True
                    left_img, right_img = create_synthetic_stereo_images()
                else:
                    # 分割左右图像
                    h, w = frame.shape[:2]
                    mid = w // 2
                    left_img = frame[:, :mid, :]
                    right_img = frame[:, mid:, :]
                    
                    if h != 800 or mid != 1280:
                        use_synthetic = True
                        left_img, right_img = create_synthetic_stereo_images()
            
            # 处理帧
            result = slam.process_frame(left_img, right_img)
            frame_count += 1
            
            # 创建显示图像 - 左右并排
            display_img = np.hstack((left_img, right_img))
            
            # 获取 3D 点并投影到左图
            positions = slam.map.get_3d_points_array()
            if len(positions) > 0:
                camera_pose = slam.get_camera_pose()
                K = slam.K.copy()
                projected = project_3d_to_2d(positions, camera_pose, K, left_img.shape)
                
                # 在左图上绘制投影点
                for pt in projected:
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= x < left_img.shape[1] and 0 <= y < left_img.shape[0]:
                        depth = pt[2] if len(pt) > 2 else 1.0
                        # 根据深度设置颜色
                        if depth < 3:
                            color = (0, 255, 0)  # 绿色 - 近
                        elif depth < 6:
                            color = (0, 255, 255)  # 黄色 - 中
                        else:
                            color = (0, 100, 255)  # 橙红色 - 远
                        cv2.circle(display_img, (x, y), 3, color, -1)
            
            # 添加文本信息
            stats = slam.get_map_statistics()
            num_points = stats.get('num_points', 0)
            cam_pos = slam.get_camera_position()
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # 在左上角显示信息
            cv2.putText(display_img, f"FPS: {fps:.1f}", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_img, f"帧数：{frame_count}", (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_img, f"3D 点数：{num_points}", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_img, f"Cam: ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f})",
                       (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 添加左右目标签
            mid_w = left_img.shape[1]
            cv2.putText(display_img, "LEFT (with 3D points)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_img, "RIGHT", (mid_w + 20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示图像
            cv2.imshow("立体摄像头 - 点云可视化 (左 | 右)", display_img)
            cv2.imshow("左目图像 (带 3D 点投影)", left_img.copy())
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] 用户提前退出")
                break
            
            # 每 5 帧输出一次日志
            if frame_count % 5 == 0:
                logger.info(f"[{elapsed:.1f}s] Frame {frame_count} | Points: {num_points}")
    
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
    
    # 最终统计
    total_time = time.time() - start_time
    stats = slam.get_map_statistics()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    print(f"总时间：{total_time:.2f}s")
    print(f"总帧数：{frame_count}")
    print(f"总 3D 点数：{stats.get('num_points', 0)}")
    print("=" * 60)
    
    return {"success": True, "total_points": stats.get('num_points', 0)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stereo SLAM Test")
    parser.add_argument("--device", type=int, default=0, help="Camera device ID")
    parser.add_argument("--duration", type=int, default=5, help="Duration in seconds")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic images")
    args = parser.parse_args()
    
    result = run_stereo_slam_test(
        device_id=args.device,
        duration_seconds=args.duration,
        use_synthetic=args.synthetic
    )
    sys.exit(0 if result["success"] else 1)