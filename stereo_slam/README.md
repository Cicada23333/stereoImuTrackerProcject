# Stereo SLAM 库

一个基于 OpenCV 的立体视觉 SLAM 系统，使用 ORB 算法进行特征提取和 3D 地图构建。

## 功能特性

- **ORB 特征提取**: 使用 ORB 算法从图像中提取特征点和描述子
- **立体匹配**: 匹配左右相机图像的特征点
- **3D 三角测量**: 通过立体匹配计算 3D 点位置，带质量检查
- **视觉里程计**: 使用 PnP 算法估计相机位姿
- **增量式地图更新**: 使用观测加权平均策略更新 3D 点位置
- **2D 投影关联**: 使用 2D 投影检查来关联地图点，比 3D 距离更可靠
- **关键帧管理**: 只在相机移动足够时才添加关键帧
- **地图管理**: 管理 3D 点和关键帧，支持点过滤
- **实时处理**: 支持实时摄像头输入和地图更新

## 模块化重构

项目已重构为模块化结构，提高了代码的可维护性和可扩展性：

### 新目录结构

```
stereo_slam/
├── src/
│   ├── __init__.py              # 主模块导出
│   ├── core/                    # 核心模块
│   │   ├── __init__.py
│   │   ├── stereo_slam.py       # 主 SLAM 系统
│   │   └── config.py            # 配置参数
│   ├── features/                # 特征处理模块
│   │   ├── __init__.py
│   │   ├── extractor.py         # ORB 特征提取
│   │   ├── matcher.py           # 特征匹配
│   │   └── tracker.py           # 特征跟踪
│   ├── geometry/                # 几何计算模块
│   │   ├── __init__.py
│   │   ├── triangulation.py     # 三角测量
│   │   ├── pose_estimation.py   # 位姿估计 (PnP)
│   │   └── utils.py             # 几何工具函数
│   ├── map/                     # 地图模块
│   │   ├── __init__.py
│   │   ├── map.py               # 地图管理
│   │   ├── keyframe.py          # 关键帧数据结构
│   │   └── point.py             # 3D 点数据结构
│   └── vo/                      # 视觉里程计模块
│       ├── __init__.py
│       ├── visual_odometry.py   # 视觉里程计
│       └── map_updater.py       # 地图更新器
├── simple_web_slam.py           # Web 可视化测试脚本
└── README.md                    # 本文档
```

### 模块说明

| 模块 | 说明 |
|------|------|
| `core` | 核心 SLAM 系统，整合所有组件 |
| `features` | 特征提取、匹配和跟踪功能 |
| `geometry` | 三角测量、位姿估计等几何计算 |
| `map` | 3D 地图和关键帧管理 |
| `vo` | 视觉里程计和地图更新 |

## 增量式地图更新（改进版）

### 改进的 3D 点更新策略

**旧方法的问题**:
- 使用简单距离阈值判断重复点 → 容易错误合并或漏合并
- 使用简单平均更新 3D 点 → 累积偏差，破坏几何一致性

**改进方法**:
- **2D 投影关联**: 将地图点投影到当前帧，检查特征点是否与投影点接近
- **观测加权平均**: 使用观测次数加权的平均策略，而不是 EMA
- **相机移动检测**: 只在相机移动足够时才更新点位置
- **质量检查**: 三角测量时检查视差和深度范围
- **点过滤**: 定期清理观测次数不足的不可靠点

### 三角测量质量检查

新点需要满足以下条件才能被接受：
- **视差范围**: 4.0 < disparity < 150.0（收紧范围以提高点质量）
- **深度范围**: 1.0m < depth < 12.0m（更合理的深度范围）

### 观测加权平均更新公式

```
P_new = (P_old * Count + P_obs) / (Count + 1)
```

其中 `Count` 是已观测次数，这样已经观测多次的点会更稳定。

### 相机移动检测

- 当相机移动超过 10cm 时，才认为移动足够添加关键帧
- 当相机没有明显移动时，只增加观测次数，不更新点位置
- 这样可以保持点的稳定性，避免累积误差

## 系统要求

- Python 3.7+
- OpenCV 4.x
- NumPy
- Flask (仅用于 Web 可视化)

## 安装

```bash
pip install opencv-python numpy flask
```

### 推荐开发环境

本仓库已补充可重复环境文件：

```powershell
cd D:\app\trackerProject\stereo_slam
.\scripts\setup_env.ps1
.\.venv\Scripts\Activate.ps1
```

依赖清单见 `requirements.txt`，完整锁定版本见 `requirements.lock.txt`。

SLAM 技术背景、OpenCV-only 稠密建图与回环闭合评估见：

```text
docs/slam_technical_background.md
```

## 使用示例

### 基本使用

```python
from src.core.stereo_slam import StereoSLAM
import cv2

# 创建 SLAM 实例。
# 默认配置匹配 2560x720 横向拼接双目图：
# 左眼 1280x720 + 右眼 1280x720，baseline=65mm，单眼水平 FOV=100°
slam = StereoSLAM(
    device_id=0,
    baseline=0.065,
    image_width=1280,
    image_height=720,
    stereo_width=2560,
    fov_horizontal=100.0,
    eye_order="left-right",
    debug_mode=True,
    auto_save_path="maps/live_map.json"
)

# 用户传入的一张 2560x720 左右拼接图
stereo_image = cv2.imread("frame_0001.png")
result = slam.process_stereo_image(stereo_image)

# 用户一次传多张图时，按顺序持续更新同一张地图
stereo_images = [cv2.imread(path) for path in image_paths]
results = slam.process_images(stereo_images, save_map_path="maps/batch_map.json")

# 也可以使用统一入口：单张返回 dict，多张返回 list[dict]
result_or_results = slam.process(stereo_image)
result_or_results = slam.process(stereo_images)
```

### 已拆分左右眼时使用

```python
left_image = stereo_image[:, :1280]
right_image = stereo_image[:, 1280:]
result = slam.process_frame(left_image, right_image)

print(f"提取了 {result['num_matches']} 个匹配点")
print(f"地图中有 {result['total_map_points']} 个 3D 点")
print(f"新增 {result['num_new_points']} 个点，更新 {result['num_updated_points']} 个点")

# 获取地图统计
stats = slam.get_map_statistics()
print(f"总 3D 点数：{stats['num_points']}")

# 保存地图
slam.save_map("my_map.json")

# 可视化地图
slam.visualize_map("map_visualization.png")
```

### 使用模块化组件

```python
# 单独使用特征提取器
from src.features import FeatureExtractor, StereoMatcher

extractor = FeatureExtractor(n_features=2000)
matcher = StereoMatcher(ratio_threshold=0.75)

left_keypoints, left_descriptors = extractor.extract(left_image)
right_keypoints, right_descriptors = extractor.extract(right_image)
matches = matcher.match_stereo_rectified(
    left_keypoints, right_keypoints,
    left_descriptors, right_descriptors
)

# 使用三角测量
from src.geometry import StereoTriangulator

triangulator = StereoTriangulator(
    baseline=0.065,
    focal_length=537.0,
    principal_point=(640.0, 360.0)
)
points_3d = triangulator.triangulate_matches(
    left_keypoints, right_keypoints, matches
)

# 使用地图管理
from src.map import Map

map = Map(device_id=0)
for feature_id, position in points_3d:
    map.add_3d_point(position=position)
```

### Device 0 摄像头调试

如果 0 号设备是 2560x720 横向拼接双目摄像头：

```powershell
cd D:\app\trackerProject\stereo_slam
.\.venv\Scripts\python.exe .\scripts\camera_debug.py --device 0 --frames 60 --eye-order right-left
```

默认会把地图保存到：

```text
.runtime/device0_map.json
```

### 命令行测试

```bash
# 使用合成图像测试
python -m pytest -q

# 使用摄像头测试
python scripts/camera_debug.py --device 0 --frames 60
```

### Web 可视化

```bash
# 启动 Web 服务器
python simple_web_slam.py

# 打开浏览器访问 http://localhost:9704
```

## 参数说明

### StereoSLAM 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| device_id | int | 0 | 摄像头设备 ID |
| baseline | float | 0.065 | 左右相机基线距离（米） |
| focal_length | float | 1000.0 | 焦距（像素） |
| image_width | int | 1280 | 单只眼图像宽度。2560x720 拼接图应为 1280 |
| image_height | int | 720 | 图像高度 |
| stereo_width | int | 2560 | 左右拼接图总宽度 |
| fov_horizontal | float | 100.0 | 水平视场角（度） |
| eye_order | str | left-right | 拼接顺序，可选 `left-right` 或 `right-left`。当前 device 0 调试建议 `right-left` |
| auto_save_path | str/path/None | None | 高层处理入口完成后自动保存地图 |
| debug_mode | bool | False | 是否启用调试模式 |

### 处理结果返回

| 字段 | 类型 | 说明 |
|------|------|------|
| frame_id | int | 帧 ID |
| success | bool | 是否处理成功 |
| num_keypoints_left | int | 左图特征点数量 |
| num_keypoints_right | int | 右图特征点数量 |
| num_matches | int | 立体匹配数量 |
| num_new_points | int | 新增 3D 点数量 |
| num_updated_points | int | 更新 3D 点数量 |
| total_map_points | int | 地图总 3D 点数 |
| camera_pose | list | 相机位姿矩阵 |
| vo_matches | int | 视觉里程计匹配数 |
| vo_inliers | int | 视觉里程计内点数 |
| timestamp | str | 时间戳 |

## 算法说明

### ORB 特征提取

使用 OpenCV 的 ORB (Oriented FAST and Rotated BRIEF) 算法提取特征点。

### 立体匹配

使用暴力匹配器 (BFMatcher) 进行特征点匹配，并应用 Lowe's ratio test 过滤低质量匹配。

### 3D 三角测量

通过立体匹配计算视差，使用三角测量公式计算 3D 点位置：

```
Z = f * B / d
X = (u - cx) * Z / f
Y = (v - cy) * Z / f
```

### 视觉里程计 (PnP)

使用 `cv2.solvePnPRansac` 估计相机位姿：
1. 匹配当前帧特征与前一帧特征
2. 使用对应的 3D 点和 2D 点坐标求解相机位姿
3. 累积相机位姿变化

### 增量式地图更新

1. **特征匹配**: 将当前帧特征与已有地图点匹配
2. **位姿估计**: 使用 PnP 估计当前相机位置
3. **三角测量**: 对立体匹配的点进行三角测量得到 3D 位置（带质量检查）
4. **2D 投影关联**: 将地图点投影到当前帧，检查新特征是否与投影点接近
5. **相机移动检测**: 检测相机是否移动超过阈值
6. **更新/添加**: 只有当相机移动时才使用加权平均更新位置，否则只增加观测次数
7. **点过滤**: 定期清理观测次数不足的不可靠点

## 配置参数

### MapConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| distance_threshold | 0.05 | 距离阈值 (5cm，防止不同物体合并) |
| max_observation_distance | 0.15 | 最大观测距离 (15cm) |
| min_observations | 3 | 最小观测次数 (需要更多观测才被认为是可靠点) |
| min_disparity | 2.0 | 最小视差 (像素) |
| max_disparity | 300.0 | 最大视差 (允许室内近距离特征) |
| min_depth | 0.25 | 最小深度 (米) |
| max_depth | 12.0 | 最大深度 (12 米，超出此距离基线不够可靠) |
| update_weight | 0.05 | 新观测的权重 (更保守) |
| depth_variance_threshold | 0.05 | 深度方差阈值 |
| min_stereo_baseline | 0.02 | 最小立体基线变化 |
| min_reprojection_error | 0.5 | 最大重投影误差 |
| min_parallax_angle | 5.0 | 最小视差角 (度) |
| min_keyframe_distance | 0.1 | 最小相机移动距离 (10cm) 才添加关键帧 |
| min_keyframe_angle | 10.0 | 最小旋转角度 (度) 才添加关键帧 |
| max_reprojection_pixel_error | 3.0 | 最大重投影像素误差 (用于 2D 关联) |
| max_vertical_disparity | 20.0 | 左右匹配点最大垂直偏差，适配未严格校正的 USB 双目输出 |

## 调试信息

启用 debug_mode 后，系统会输出详细的调试信息，包括：
- 特征点提取数量
- 立体匹配数量
- 视觉里程计内点数
- 相机移动距离
- 三角测量结果
- 新增/更新点数
- 地图更新状态

## 版本历史

### v2.3.0 (当前版本)
- 使用 2D 投影关联代替 3D 距离检查
- 只在相机移动足够时才添加关键帧
- 使用观测加权平均而不是 EMA
- 相机静止时只增加观测次数，不更新位置
- 收紧三角测量条件（min_disparity=4.0, max_depth=12.0）
- 降低距离阈值到 5cm，防止不同物体合并
- 降低 update_weight 到 0.05（更保守）

### v2.2.0
- 收紧三角测量条件以提高点质量
- 提高最小视差从 0.5 到 2.0
- 降低最大视差从 1000.0 到 200.0
- 提高最小深度从 0.2m 到 1.0m
- 降低最大深度从 50.0m 到 15.0m
- 增加最小观测次数从 1 到 3
- 降低 update_weight 从 0.1 到 0.05（更保守）

### v2.1.0
- 放松三角测量条件，增加特征点数量
- 改进 3D 点更新策略（更保守的权重）
- 添加相机移动检测机制
- 只在相机移动时更新 3D 点位置
- 添加深度稳定性配置

### v2.0.0
- 重构为模块化结构
- 新增配置管理模块
- 分离几何计算功能
- 改进 3D 点更新策略（加权平均）
- 添加三角测量质量检查
- 支持点过滤和清理

### v1.0.0
- 初始版本
- 基本 SLAM 功能实现

## 许可证

MIT License
