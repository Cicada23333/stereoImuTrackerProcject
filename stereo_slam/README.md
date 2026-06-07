# Stereo SLAM

基于 OpenCV + ORB 的双目 SLAM 原型库。库的主要输入是一张横向拼接双目图，默认规格为：

- 整图：`2560x720`
- 左右眼：每只眼 `1280x720`
- 双目基线：`65mm`
- 单眼水平 FOV：`100°`

当前版本支持单张图、批量图、文件路径和 numpy 数组输入。每处理一帧都会提取 ORB 特征、做双目匹配、三角化当前观测、估计相机运动，并增量更新 3D 地图。

## 当前能力

- 处理 `2560x720` 横向拼接双目图。
- 支持 `left-right` 和 `right-left` 两种拼接顺序。
- 使用 ORB + Hamming 匹配提取左右眼特征。
- 使用视差、深度和垂直偏差过滤低质量匹配。
- 使用 PnP RANSAC 做视觉里程计。
- 将当前相机坐标中的三角化点转换到世界坐标后写入地图。
- 使用 2D 投影关联更新已有地图点。
- 保存和加载 JSON 地图。
- Web 调试页显示当前帧成功三角化的双目观测点，便于确认点是否贴在图像特征上。

## 限制

这是 OpenCV-only 原型，不是完整生产级 SLAM：

- 没有双目标定流程，默认使用 FOV 推导焦距。
- 没有全局 bundle adjustment 或位姿图优化。
- 没有回环检测/闭环优化。
- 真实深度和长期轨迹会受未标定畸变、未校正双目、弱纹理和曝光影响。

如果要让地图长期稳定，下一步应加入双目标定、矫正参数加载、后端优化和回环。

## 项目结构

```text
stereo_slam/
├── simple_web_slam.py              # Web 调试兼容入口
├── requirements.txt
├── requirements.lock.txt
├── docs/
│   └── slam_technical_background.md
├── scripts/
│   ├── camera_debug.py             # device 0 摄像头调试
│   └── setup_env.ps1               # Windows 环境初始化
├── src/
│   ├── core/
│   │   ├── stereo_slam.py          # StereoSLAM 主类装配
│   │   ├── image_input.py          # 拼接图输入、批量输入、自动保存
│   │   ├── frame_processing.py     # 每帧 SLAM 管线
│   │   ├── observations.py         # 当前帧三角化观测
│   │   ├── map_association.py      # 地图点投影关联和更新
│   │   ├── pose.py                 # 坐标系和相机位姿工具
│   │   ├── map_io.py               # 地图保存、加载、可视化
│   │   └── config.py               # 配置参数
│   ├── features/
│   │   ├── extractor.py            # ORB 特征提取
│   │   ├── matcher.py              # 双目匹配和过滤
│   │   └── tracker.py
│   ├── geometry/
│   │   ├── triangulation.py
│   │   ├── pose_estimation.py
│   │   └── utils.py                # 投影和几何工具
│   ├── map/
│   │   ├── map.py
│   │   ├── point.py
│   │   └── keyframe.py
│   ├── vo/
│   │   ├── visual_odometry.py
│   │   └── map_updater.py
│   └── web/
│       ├── app.py                  # Flask app 和后台处理线程
│       ├── camera.py               # 摄像头打开和帧裁剪
│       ├── synthetic.py            # 合成双目帧
│       └── templates.py            # Web 页面模板
└── tests/
    └── test_synthetic_slam.py
```

## 安装

推荐使用项目内虚拟环境：

```powershell
cd D:\app\trackerProject\stereo_slam
.\scripts\setup_env.ps1
.\.venv\Scripts\Activate.ps1
```

也可以手动安装：

```powershell
pip install -r requirements.txt
```

## 基本 API

### 单张拼接双目图

```python
import cv2
from src.core.stereo_slam import StereoSLAM

slam = StereoSLAM(
    baseline=0.065,
    image_width=1280,
    image_height=720,
    stereo_width=2560,
    fov_horizontal=100.0,
    eye_order="left-right",
    auto_save_path="maps/live_map.json",
)

frame = cv2.imread("frame_0001.png")  # 2560x720 side-by-side stereo frame
result = slam.process_stereo_image(frame)

print(result["num_matches"])
print(result["num_current_observations"])
print(result["total_map_points"])
```

### 多张图连续更新同一张地图

```python
frames = [cv2.imread(path) for path in image_paths]
results = slam.process_images(frames, save_map_path="maps/batch_map.json")
```

### 统一入口

```python
single_result = slam.process(frame)
batch_results = slam.process(frames)
```

`process()` 接收：

- 单张 numpy 图像
- 图片路径
- 图片 list/tuple
- 4D numpy batch，例如 `(N, 720, 2560, 3)`

### 已拆分左右眼图像

```python
left_image, right_image = slam.split_stereo_image(frame)
result = slam.process_frame(left_image, right_image)
```

## 返回字段

常用结果字段：

| 字段 | 说明 |
|---|---|
| `success` | 当前帧是否处理成功 |
| `frame_id` | 帧号，自动递增 |
| `num_keypoints_left` | 左眼 ORB 特征数 |
| `num_keypoints_right` | 右眼 ORB 特征数 |
| `num_matches` | 双目匹配数 |
| `num_triangulated_points` | 通过质量检查的三角化点数 |
| `num_current_observations` | 当前帧成功三角化并用于 Web overlay 的观测点数 |
| `num_new_points` | 新增地图点 |
| `num_updated_points` | 更新地图点 |
| `total_map_points` | 当前地图点总数 |
| `camera_pose` | world-to-camera 位姿矩阵 |
| `vo_matches` | VO 匹配数 |
| `vo_inliers` | PnP RANSAC 内点数 |

## Device 0 摄像头调试

当前测试环境的 device 0 输出实际是 `2560x800`，调试脚本会居中裁剪到 `2560x720`。实测该摄像头拼接顺序是 `right-left`。

```powershell
cd D:\app\trackerProject\stereo_slam
.\.venv\Scripts\python.exe .\scripts\camera_debug.py --device 0 --frames 60 --eye-order right-left
```

默认地图输出：

```text
.runtime/device0_map.json
```

## Web 调试

```powershell
cd D:\app\trackerProject\stereo_slam
.\.venv\Scripts\python.exe .\simple_web_slam.py
```

打开：

```text
http://localhost:9704
```

Web 调试默认会自动保存地图：

```text
.runtime/web_live_map.json
```

默认每处理 30 帧保存一次，按 `Ctrl+C` 退出时还会再保存一次。也可以指定输出路径：

```powershell
.\.venv\Scripts\python.exe .\simple_web_slam.py --save-map .runtime\my_web_map.json --save-every 15
```

如果只想退出时保存一次：

```powershell
.\.venv\Scripts\python.exe .\simple_web_slam.py --save-every 0
```

如果不想保存：

```powershell
.\.venv\Scripts\python.exe .\simple_web_slam.py --no-save-map
```

Web 页面画的是当前帧成功三角化的双目观测点：

- 左眼：绿色/黄色/橙色
- 右眼：蓝色/青色/浅蓝色
- 颜色按深度分段

这不是历史地图点投影。这样做是为了调试“点是否贴在当前图像特征上”。历史地图仍会在后台更新。

## 关键配置

`src/core/config.py` 里的当前默认值偏向室内近距离调试：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `min_disparity` | `2.0` | 最小视差 |
| `max_disparity` | `300.0` | 最大视差，允许近距离室内特征 |
| `min_depth` | `0.25` | 最小深度，单位米 |
| `max_depth` | `12.0` | 最大深度，单位米 |
| `max_vertical_disparity` | `20.0` | 左右匹配点最大 y 偏差，适配未严格校正的 USB 双目 |
| `max_reprojection_pixel_error` | `3.0` | 2D 投影关联搜索半径 |
| `min_keyframe_distance` | `0.1` | 添加关键帧的最小移动距离，单位米 |

## 测试

```powershell
cd D:\app\trackerProject\stereo_slam
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

当前测试覆盖：

- 合成双目帧处理
- 帧号递增
- 拼接图拆分
- 单张/批量 API
- 地图保存与加载
- 当前观测点生成

## 最近重要修复

- 修复地图点坐标系：三角化点先从相机坐标转换到世界坐标后入图。
- 修复投影索引错位：投影过滤后保留原始地图点索引。
- 修复 VO 位姿累积顺序。
- Web overlay 改为当前帧观测点，避免把漂移的历史地图投影误认为特征绑定失败。
- 放宽室内近距离三角化门限。
- 拆分核心和 Web 大文件，降低维护成本。
