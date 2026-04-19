#!/usr/bin/env python3
"""
pyslam 实时立体摄像头追踪示例

使用 id 为 2 的摄像头（2560x720 并排双摄，间距 65mm，平行镜头，FOV 100 度）
将视频流分割成左右两个 1280x720 的图像流，进行 SLAM 追踪，并显示追踪图和特征图。

使用方法:
    python stereo_camera_slam.py           # 简化版特征追踪
    python stereo_camera_slam.py --full    # 完整 pyslam SLAM (需要安装 pyslam)
    python stereo_camera_slam.py --help    # 显示帮助

注意:
    对于完整的 pyslam 功能，需要从 GitHub 克隆并安装:
    git clone --recursive https://github.com/luigifreda/pyslam.git
    cd pyslam
    ./install_all.sh
"""

import cv2
import numpy as np
import time
import sys
import os

# 添加 pyslam 路径（如果已安装）
HAS_PYSLAM = False
try:
    from pyslam.config import Config
    from pyslam.slam.camera import PinholeCamera
    from pyslam.io.dataset import Dataset
    from pyslam.io.dataset_types import SensorType
    from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
    from pyslam.loop_closing.loop_detector_configs import LoopDetectorConfigs
    from pyslam.slam.slam import Slam, SlamState
    from pyslam.viz.slam_plot_drawer import SlamPlotDrawerThread
    from pyslam.viz.viewer3D import Viewer3D
    from pyslam.utilities.colors import GlColors
    from pyslam.utilities.img_management import ImgWriter
    from pyslam.utilities.logging import Printer
    from pyslam.utilities.timer import TimerFps
    from pyslam.viz.cvimage_thread import CvImageViewer
    HAS_PYSLAM = True
except ImportError:
    print("警告：无法导入 pyslam。将使用简化版 SLAM 演示。")
    print("安装完整 pyslam: git clone --recursive https://github.com/luigifreda/pyslam.git")


class StereoSplitter:
    """
    双目摄像头分割器
    
    将 2560x720 的并排双目图像分割成左右两个独立的 1280x720 图像流
    """
    
    # 默认参数
    DEFAULT_WIDTH = 2560      # 双目总宽度
    DEFAULT_HEIGHT = 720      # 图像高度
    DEFAULT_BASELINE = 0.065  # 基线距离 (米)
    
    def __init__(
        self,
        camera_index: int = 0,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        baseline: float = DEFAULT_BASELINE,
        backend: str = "msmf"
    ):
        """
        初始化分割器
        
        Args:
            camera_index: 摄像头设备索引
            width: 双目图像总宽度
            height: 图像高度
            baseline: 双目基线距离 (米)
            backend: OpenCV 后端 ("msmf" 或 "dshow")
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.baseline = baseline
        
        # 计算左右图像的尺寸
        self.single_width = width // 2  # 1280
        self.single_height = height     # 720
        
        # 摄像头捕获对象
        self.cap = None
        self.is_running = False
        
        # 相机内参矩阵
        self._init_intrinsic_matrix()
        
        # 根据后端选择 API
        if backend == "dshow":
            self.backend = cv2.CAP_DSHOW
        else:
            self.backend = cv2.CAP_MSMF
    
    def _init_intrinsic_matrix(self):
        """
        初始化相机内参矩阵
        
        基于 100 度 FOV 和 1280x720 分辨率计算
        """
        # 根据 FOV 计算焦距
        fov_rad = np.radians(100)  # 100 度水平 FOV
        fx = self.single_width / (2 * np.tan(fov_rad / 2))
        fy = fx  # 假设正方形像素
        
        # 光心在单目图像中心
        cx = self.single_width / 2
        cy = self.single_height / 2
        
        # 内参矩阵
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # 畸变系数 (初始化为零，需要根据实际标定调整)
        self.dist_coeffs = np.zeros(5)
        
        # bf 值 (baseline * fx) - 用于深度计算
        self.bf = self.baseline * 1000 * fx  # baseline 转换为毫米
        
        print(f"相机内参矩阵 K:")
        print(self.K)
        print(f"畸变系数：{self.dist_coeffs}")
        print(f"bf 值 (baseline*fx): {self.bf:.2f}")
    
    def open(self) -> bool:
        """
        打开摄像头
        
        Returns:
            是否成功打开
        """
        print(f"[DEBUG] 尝试打开摄像头 (索引：{self.camera_index})...")
        self.cap = cv2.VideoCapture(self.camera_index, self.backend)
        
        if not self.cap.isOpened():
            print(f"错误：无法打开摄像头 (索引：{self.camera_index})")
            return False
        
        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # 验证设置
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"摄像头实际分辨率：{actual_width}x{actual_height}")
        print(f"摄像头实际帧率：{actual_fps:.1f} FPS")
        
        if actual_width != self.width or actual_height != self.height:
            print(f"警告：期望分辨率 {self.width}x{self.height}，实际为 {actual_width}x{actual_height}")
            # 更新实际分辨率
            self.width = actual_width
            self.height = actual_height
            self.single_width = self.width // 2
            # 重新初始化内参
            self._init_intrinsic_matrix()
        
        self.is_running = True
        return True
    
    def split_frame(self, frame: np.ndarray):
        """
        分割帧为左右两个图像
        
        Args:
            frame: 输入的双目图像 (2560x720 并排格式)
        
        Returns:
            (左目图像，右目图像)
        """
        # 确保帧尺寸正确
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            # 调整帧尺寸
            frame = cv2.resize(frame, (self.width, self.height))
        
        # 分割左右图像
        left_frame = frame[:, :self.single_width].copy()
        right_frame = frame[:, self.single_width:].copy()
        
        return left_frame, right_frame
    
    def read(self):
        """
        读取并分割一帧
        
        Returns:
            (左目图像，右目图像，成功标志)
        """
        if not self.is_running or self.cap is None:
            return None, None, False
        
        ret, frame = self.cap.read()
        
        if not ret or frame is None:
            return None, None, False
        
        left_frame, right_frame = self.split_frame(frame)
        
        return left_frame, right_frame, True
    
    def get_camera_info(self) -> dict:
        """
        获取相机信息
        
        Returns:
            相机参数字典
        """
        return {
            'width': self.width,
            'height': self.height,
            'single_width': self.single_width,
            'single_height': self.single_height,
            'baseline': self.baseline,
            'K': self.K.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
            'bf': self.bf
        }
    
    def release(self):
        """释放摄像头资源"""
        if self.cap:
            self.cap.release()
            self.is_running = False


def estimate_camera_intrinsics(width, height, fov_deg, baseline_mm):
    """
    根据 FOV 和分辨率估算摄像头内参
    
    对于平行立体摄像头系统:
    - 焦距计算公式：fx = width / (2 * tan(fov/2))
    - 主点位于图像中心
    - bf = baseline * fx (用于深度计算，pyslam 要求单位为像素*毫米)
    
    参数:
        width: 单目图像宽度
        height: 图像高度
        fov_deg: 水平视场角 (度)
        baseline_mm: 立体基线距离 (mm)
    
    返回:
        fx, fy, cx, cy, k1, k2, p1, p2, k3, bf
    """
    # 计算焦距 (使用水平 FOV)
    fov_rad = np.radians(fov_deg)
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = fx  # 假设正方形像素
    
    # 主点 (图像中心)
    cx = width / 2
    cy = height / 2
    
    # 简化畸变系数 (实际使用时需要校准)
    # 对于广角镜头 (100 度 FOV)，需要一定的畸变校正
    k1 = -0.2   # 径向畸变
    k2 = 0.08   # 径向畸变
    p1 = 0.0    # 切向畸变
    p2 = 0.0    # 切向畸变
    k3 = 0.0    # 径向畸变
    
    # bf 值 (baseline * fx) - 用于深度计算
    # pyslam 使用 baseline * fx 作为 bf 参数，单位为像素*毫米
    bf = baseline_mm * fx
    
    print(f"[DEBUG] 摄像头内参计算:")
    print(f"[DEBUG]   单目分辨率：{width} x {height}")
    print(f"[DEBUG]   FOV: {fov_deg}度")
    print(f"[DEBUG]   基线：{baseline_mm}mm")
    print(f"[DEBUG]   fx: {fx:.2f} pixels")
    print(f"[DEBUG]   fy: {fy:.2f} pixels")
    print(f"[DEBUG]   cx: {cx:.2f} pixels")
    print(f"[DEBUG]   cy: {cy:.2f} pixels")
    print(f"[DEBUG]   bf: {bf:.2f}")
    
    return fx, fy, cx, cy, k1, k2, p1, p2, k3, bf


def simple_feature_tracking_demo():
    """
    简化版立体摄像头特征追踪演示 (不依赖完整 pyslam)
    
    使用 OpenCV 的 ORB 特征检测和光流追踪
    显示捕获的图像和特征地图
    """
    print("=" * 60)
    print("pyslam 简化版立体摄像头特征追踪演示")
    print("=" * 60)
    
    # 摄像头配置 - 根据用户要求
    CAMERA_ID = 0  # 并排立体摄像头 ID
    TOTAL_WIDTH = 2560  # 双目总宽度
    HEIGHT = 720
    SINGLE_WIDTH = TOTAL_WIDTH // 2  # 1280
    FOV_DEG = 100  # 水平 FOV
    BASELINE_MM = 65  # 基线距离 (mm)
    
    # DEBUG 信息
    print(f"\n[DEBUG] Python 版本：{sys.version.split()[0]}")
    print(f"[DEBUG] OpenCV 版本：{cv2.__version__}")
    print(f"[DEBUG] NumPy 版本：{np.__version__}")
    
    # 估算摄像头内参
    fx, fy, cx, cy, k1, k2, p1, p2, k3, bf = estimate_camera_intrinsics(
        SINGLE_WIDTH, HEIGHT, FOV_DEG, BASELINE_MM
    )
    
    print(f"\n[INFO] 摄像头内参估算:")
    print(f"[INFO]   焦距 fx: {fx:.2f}")
    print(f"[INFO]   焦距 fy: {fy:.2f}")
    print(f"[INFO]   主点 cx: {cx:.2f}")
    print(f"[INFO]   主点 cy: {cy:.2f}")
    print(f"[INFO]   基线×焦距 (bf): {bf:.2f}")
    
    # 创建分割器
    print(f"\n[DEBUG] 创建立体摄像头分割器...")
    splitter = StereoSplitter(
        camera_index=CAMERA_ID,
        width=TOTAL_WIDTH,
        height=HEIGHT,
        baseline=BASELINE_MM / 1000  # 转换为米
    )
    
    # 打开摄像头
    if not splitter.open():
        print("[ERROR] 无法打开摄像头")
        print("[ERROR] 提示：请检查摄像头连接并确认摄像头 ID 是否正确")
        return
    
    # 获取实际分辨率
    actual_width = splitter.width
    actual_height = splitter.height
    actual_single_width = splitter.single_width
    
    print(f"\n[INFO] 实际分辨率：{actual_width} x {actual_height} (并排)")
    print(f"[INFO] 单目分辨率：{actual_single_width} x {actual_height}")
    
    # 初始化 ORB 特征检测器
    print(f"[DEBUG] 初始化 ORB 特征检测器...")
    orb = cv2.ORB_create(nfeatures=500, fastThreshold=20)
    
    # 初始化特征追踪器 (KLT 光流)
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03)
    )
    
    # 存储上一帧的特征点
    old_left_points = None
    old_left_gray = None
    
    # 创建随机颜色用于绘制轨迹
    np.random.seed(42)
    color = np.random.randint(0, 255, (500, 3), dtype=np.uint8)
    
    # 帧计数器
    frame_count = 0
    start_time = time.time()
    
    print(f"\n[INFO] 开始立体特征追踪...")
    print(f"[INFO] 按 'q' 退出，按 'r' 重置追踪")
    
    try:
        while True:
            # 读取并分割帧
            left_frame, right_frame, success = splitter.read()
            
            if not success:
                print(f"[ERROR] 无法读取摄像头帧")
                break
            
            frame_count += 1
            
            # 转换为灰度图
            left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
            
            # 创建结果图像 - 并排显示左右目
            img_display = np.hstack((left_frame.copy(), right_frame.copy()))
            
            if old_left_gray is None:
                # 第一帧：检测 ORB 特征点
                print(f"[DEBUG] 第一帧：检测 ORB 特征点...")
                keypoints = orb.detect(left_gray, None)
                
                if keypoints:
                    # 转换为光流格式 (N, 1, 2) 的 float32 数组
                    old_left_points = np.float32([kp.pt for kp in keypoints])
                    old_left_points = old_left_points.reshape(-1, 1, 2)
                    print(f"[DEBUG] 左目检测到 {len(keypoints)} 个 ORB 特征点")
                else:
                    # 备用方案：使用 goodFeaturesToTrack
                    old_left_points = cv2.goodFeaturesToTrack(
                        left_gray,
                        maxCorners=500,
                        qualityLevel=0.01,
                        minDistance=20
                    )
                    if old_left_points is not None:
                        print(f"[DEBUG] 左目使用 goodFeaturesToTrack 检测到 {len(old_left_points)} 个特征点")
                
                old_left_gray = left_gray.copy()
                
            else:
                # 追踪左目特征点
                if old_left_points is not None and len(old_left_points) > 0:
                    try:
                        # 计算光流
                        new_points, status, err = cv2.calcOpticalFlowPyrLK(
                            old_left_gray, left_gray, old_left_points, None, **lk_params
                        )
                        
                        # 选择好的追踪点
                        if new_points is not None and status is not None:
                            good_mask = status.flatten() == 1
                            good_new = new_points[good_mask]
                            good_old = old_left_points[good_mask]
                            
                            print(f"[DEBUG] 追踪结果：成功 {len(good_new)} / {len(old_left_points)} 点")
                            
                            # 绘制轨迹 (只在左目图像上)
                            for i, (new, old) in enumerate(zip(good_new, good_old)):
                                if i >= len(color):
                                    break
                                # new 和 old 的形状为 (1, 2)，需要正确访问坐标
                                a = int(new[0][0])  # x 坐标
                                b = int(new[0][1])  # y 坐标
                                c = int(old[0][0])  # x 坐标
                                d = int(old[0][1])  # y 坐标
                                
                                # 绘制轨迹线
                                cv2.line(img_display, (a, b), (c, d), color[i % len(color)].tolist(), 1)
                                # 绘制当前点
                                cv2.circle(img_display, (a, b), 4, color[i % len(color)].tolist(), -1)
                            
                            # 更新旧点 - 保持 (N, 1, 2) 格式
                            old_left_points = good_new.reshape(-1, 1, 2)
                            old_left_gray = left_gray.copy()
                            
                    except Exception as e:
                        print(f"[ERROR] 光流计算错误：{e}")
                        import traceback
                        traceback.print_exc()
                        # 重置
                        old_left_points = None
                        old_left_gray = left_gray.copy()
            
            # 添加文本信息
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            info_text = [
                f"FPS: {fps:.1f}",
                f"帧数：{frame_count}",
                f"特征点：{len(old_left_points) if old_left_points is not None else 0}",
                f"分辨率：{actual_single_width} x {actual_height} (单目)",
            ]
            
            y_pos = 30
            for text in info_text:
                cv2.putText(img_display, text, (20, y_pos),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_pos += 40
            
            # 添加左右目标签
            cv2.putText(img_display, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img_display, "RIGHT", (actual_single_width + 20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示图像
            cv2.imshow("立体摄像头 - 特征追踪 (左 | 右)", img_display)
            cv2.imshow("左目灰度图", left_gray)
            cv2.imshow("右目灰度图", right_gray)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] 用户提前退出")
                break
            elif key == ord('r'):
                print("[INFO] 重置特征追踪")
                old_left_points = None
                old_left_gray = None
    
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断 (Ctrl+C)")
    
    finally:
        # 最终统计
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"[DEBUG] === 最终统计 ===")
        print(f"[DEBUG] 总运行时间：{total_time:.2f} 秒")
        print(f"[DEBUG] 总处理帧数：{frame_count}")
        if total_time > 0:
            print(f"[DEBUG] 平均 FPS: {frame_count / total_time:.2f}")
        print(f"[DEBUG] 左目特征点数量：{len(old_left_points) if old_left_points is not None else 0}")
        print(f"{'='*60}\n")
        
        # 清理
        splitter.release()
        cv2.destroyAllWindows()
        print("[INFO] 演示结束")


def full_pyslam_stereo_demo():
    """
    完整 pyslam 立体 SLAM 演示
    
    使用 pyslam 的完整 SLAM 功能进行立体摄像头追踪
    """
    if not HAS_PYSLAM:
        print("错误：pyslam 未正确安装")
        print("请从 GitHub 安装完整版本:")
        print("  git clone --recursive https://github.com/luigifreda/pyslam.git")
        print("  cd pyslam")
        print("  ./install_all.sh")
        print("\n或者使用简化版：python stereo_camera_slam.py")
        return
    
    print("=" * 60)
    print("pyslam 完整立体 SLAM 演示")
    print("=" * 60)
    
    # 摄像头配置
    CAMERA_ID = 0
    TOTAL_WIDTH = 2560
    HEIGHT = 720
    SINGLE_WIDTH = TOTAL_WIDTH // 2
    FOV_DEG = 100
    BASELINE_MM = 65
    
    # 估算摄像头内参
    fx, fy, cx, cy, k1, k2, p1, p2, k3, bf = estimate_camera_intrinsics(
        SINGLE_WIDTH, HEIGHT, FOV_DEG, BASELINE_MM
    )
    
    print(f"\n摄像头内参:")
    print(f"  fx: {fx:.2f}, fy: {fy:.2f}")
    print(f"  cx: {cx:.2f}, cy: {cy:.2f}")
    print(f"  bf: {bf:.2f}")
    
    # 创建摄像头设置
    cam_settings = {
        "Camera.fx": float(fx),
        "Camera.fy": float(fy),
        "Camera.cx": float(cx),
        "Camera.cy": float(cy),
        "Camera.k1": float(k1),
        "Camera.k2": float(k2),
        "Camera.p1": float(p1),
        "Camera.p2": float(p2),
        "Camera.k3": float(k3),
        "Camera.width": int(SINGLE_WIDTH),
        "Camera.height": int(HEIGHT),
        "Camera.bf": float(bf),
        "Camera.fps": 30.0,
        "Camera.RGB": 1,
        "ThDepth": 40.0,
        "DepthMapFactor": 1.0,
    }
    
    # 创建分割器
    print("\n初始化立体摄像头...")
    splitter = StereoSplitter(
        camera_index=CAMERA_ID,
        width=TOTAL_WIDTH,
        height=HEIGHT,
        baseline=BASELINE_MM / 1000
    )
    
    if not splitter.open():
        print("错误：无法打开摄像头")
        return
    
    # 创建摄像头对象
    camera = PinholeCamera.from_settings(cam_settings)
    
    # 选择特征追踪配置
    feature_tracker_config = FeatureTrackerConfigs.ORB2
    
    # 选择循环检测配置
    loop_detection_config = LoopDetectorConfigs.DBOW3
    
    print("\n初始化 SLAM 系统...")
    
    # 创建 SLAM 对象
    slam = Slam(
        camera=camera,
        feature_tracker_config=feature_tracker_config,
        loop_detection_config=loop_detection_config,
        semantic_mapping_config=None,
        sensor_type=SensorType.STEREO,
        environment_type=None,
        config=None,
        headless=False
    )
    
    # 创建 3D 查看器
    viewer3D = Viewer3D(scale=0.1)
    viewer3D.wait_for_ready()
    
    print("\n开始 SLAM 追踪... 按 'q' 退出")
    print("按 's' 保存地图，按 'r' 重置")
    
    frame_id = 0
    start_time = time.time()
    
    try:
        while True:
            # 获取图像
            left_frame, right_frame, success = splitter.read()
            
            if not success:
                time.sleep(0.1)
                continue
            
            timestamp = time.time()
            
            # SLAM 追踪
            slam.track(left_frame, right_frame, None, frame_id, timestamp)
            
            # 绘制特征轨迹
            img_draw = slam.map.draw_feature_trails(
                left_frame,
                with_level_radius=False,
                trail_max_length=100
            )
            
            # 添加文本信息
            fps = frame_id / (time.time() - start_time) if frame_id > 0 else 0
            info_text = f"FPS: {fps:.1f} | 帧：{frame_id}"
            cv2.putText(img_draw, info_text, (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示图像
            cv2.imshow("SLAM 追踪", img_draw)
            cv2.imshow("左目图像", left_frame)
            
            # 更新 3D 地图
            viewer3D.draw_slam_map(slam)
            
            frame_id += 1
            
            # 检查退出条件
            if viewer3D.is_closed():
                break
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                slam.save_system_state("./slam_state")
                print("地图已保存")
            elif key == ord('r'):
                slam.reset()
                print("SLAM 已重置")
    
    except KeyboardInterrupt:
        print("\n用户中断")
    
    finally:
        # 清理资源
        slam.quit()
        viewer3D.quit()
        splitter.release()
        
        print("SLAM 演示结束")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("pyslam 立体摄像头 SLAM 示例")
    print("=" * 60)
    print("\n摄像头配置:")
    print("  - 摄像头 ID: 0 (并排立体摄像头)")
    print("  - 总分辨率：2560 x 720 (并排)")
    print("  - 单目分辨率：1280 x 720")
    print("  - 基线距离：65mm")
    print("  - FOV: 100 度")
    print("\n" + "=" * 60)
    
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--full":
            # 运行完整 pyslam 版本
            if HAS_PYSLAM:
                full_pyslam_stereo_demo()
            else:
                print("错误：完整模式需要安装 pyslam")
                print("运行：git clone --recursive https://github.com/luigifreda/pyslam.git")
                return
        elif sys.argv[1] == "--help":
            print("\n用法:")
            print("  python stereo_camera_slam.py           # 简化版特征追踪")
            print("  python stereo_camera_slam.py --full    # 完整 pyslam SLAM")
            print("  python stereo_camera_slam.py --help    # 显示帮助")
            print("\n注意:")
            print("  - 简化版使用 OpenCV 的 ORB 特征和光流追踪")
            print("  - 完整版需要安装 pyslam GitHub 版本")
            print("  - 摄像头 ID 默认为 0 (并排立体摄像头)")
            return
    
    # 默认运行简化版
    simple_feature_tracking_demo()


if __name__ == "__main__":
    main()