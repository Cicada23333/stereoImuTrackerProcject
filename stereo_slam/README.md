# Stereo SLAM 库

一个基于 OpenCV 的立体视觉 SLAM 系统，使用 ORB 算法进行特征提取和 3D 地图构建。

## 功能特性

- **ORB 特征提取**: 使用 ORB 算法从图像中提取特征点和描述子
- **立体匹配**: 匹配左右相机图像的特征点
- **3D 三角测量**: 通过立体匹配计算 3D 点位置，带质量检查
- **视觉里程计**: 使用 PnP 算法估计相机位姿
- **增量式地图更新**: 使用加权平均策略更新 3D 点位置，只在相机移动时更新
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
- **加权平均更新**: 使用指数移动平均 (EMA) 更新 3D 点位置
- **相机移动检测**: 只在相机移动超过阈值时才更新点位置
- **质量检查**: 三角测量时检查视差和深度范围
- **点过滤**: 定期清理观测次数不足的不可靠点

### 三角测量质量检查

新点需要满足以下条件才能被接受：
- **视差范围**: 2.0 < disparity < 200.0（收紧范围以提高点质量）
- **深度范围**: 1.0m < depth < 15.0m（更合理的深度范围）

### 加权平均更新公式

```
alpha = weight / (old_weight + weight + 1e-8)
new_position = old_position * (1 - alpha) + new_observation * alpha
```

其中 `weight` 默认为 0.05，确保新观测对位置的影响逐渐减小。

### 相机移动检测

- 当相机移动超过 1cm 时，才更新 3D 点位置
- 当相机静止时，只增加观测次数，不更新位置
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

## 使用示例

### 基本使用

```python
from src.core.stereo_slam import StereoSLAM
import cv2

# 创建 SLAM 实例
slam = StereoSLAM(
    device_id=0,
    baseline=0.065,  # 65mm 基线
    image_width=2560,
    image_height=720,
    fov_horizontal=100.0,
    debug_mode=True
)

# 处理图像帧
left_image = ...  # 左眼图像
right_image = ...  # 右眼图像

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
    focal_length=1000.0,
    principal_point=(1280.0, 360.0)
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

### 命令行测试

```bash
# 使用摄像头测试（5 秒）
python test_stereo_slam.py --device 0 --duration 5

# 使用合成图像测试
python test_stereo_slam.py --synthetic --duration 5
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
| image_width | int | 2560 | 图像宽度 |
| image_height | int | 720 | 图像高度 |
| fov_horizontal | float | 100.0 | 水平视场角（度） |
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
4. **点关联**: 检查新 3D 点是否与已有地图点相近（距离阈值）
5. **相机移动检测**: 检测相机是否移动超过阈值
6. **更新/添加**: 只有当相机移动时才使用加权平均更新位置，否则只增加观测次数
7. **点过滤**: 定期清理观测次数不足的不可靠点

## 配置参数

### MapConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| distance_threshold | 0.1 | 距离阈值（用于初步筛选，更严格） |
| max_observation_distance | 0.3 | 最大观测距离（更严格） |
| min_observations | 3 | 最小观测次数（需要更多观测才被认为是可靠点） |
| min_disparity | 2.0 | 最小视差（提高以拒绝远距离/噪声点） |
| max_disparity | 200.0 | 最大视差（降低以拒绝过近/不可靠点） |
| min_depth | 1.0 | 最小深度（米，提高） |
| max_depth | 15.0 | 最大深度（米，降低） |
| update_weight | 0.05 | 新观测的权重（更保守） |
| depth_variance_threshold | 0.05 | 深度方差阈值（更严格） |
| min_stereo_baseline | 0.02 | 最小立体基线变化（提高） |
| min_reprojection_error | 0.5 | 最大重投影误差 |
| min_parallax_angle | 5.0 | 最小视差角（度） |

## 调试信息

启用 debug_mode 后，系统会输出详细的调试信息，包括：
- 特征点提取数量
- 立体匹配数量
- 视觉里程计内点数
- 三角测量结果
- 新增/更新点数
- 地图更新状态

## 版本历史

### v2.2.0 (当前版本)
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