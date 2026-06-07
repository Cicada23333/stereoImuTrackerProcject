# Stereo SLAM 库使用说明书

本文面向第一次接触本库的程序员，说明如何安装、输入什么数据、如何建图、如何保存和读取地图、如何使用已有地图定位，以及所有对外类和函数的参数含义。

## 1. 库的用途

`stereo_slam` 是一个基于 OpenCV + ORB 的双目视觉 SLAM 原型库。它默认处理一张横向拼接的双目图：

| 项目 | 默认值 | 说明 |
|---|---:|---|
| 整张输入图 | `2560x720` | 左右眼横向拼在一张图里 |
| 单眼图像 | `1280x720` | 左眼或右眼各自的尺寸 |
| 双目基线 | `0.065 m` | 两个摄像头中心距离，65 mm |
| 单眼水平 FOV | `100.0` | 用来估算焦距 |
| 默认摄像头 | `device 0` | OpenCV 摄像头 ID |
| 实测左右顺序 | `right-left` | 当前测试摄像头通常右眼在左半边，左眼在右半边 |

本库当前支持两类主要工作流：

1. 建图：持续输入双目图，提取 ORB 特征，三角化 3D 点，增量更新地图并保存 JSON。
2. 只读定位：读取已经保存的地图，用当前双目图的 ORB 特征匹配地图点 descriptor，通过 PnP RANSAC 估计相机位置，不更新地图。

## 2. 安装和运行环境

推荐在项目目录内使用虚拟环境：

```powershell
cd D:\app\trackerProject\stereo_slam
.\scripts\setup_env.ps1
.\.venv\Scripts\Activate.ps1
```

手动安装依赖：

```powershell
cd D:\app\trackerProject\stereo_slam
pip install -r requirements.txt
```

运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

## 3. 快速开始

### 3.1 用单张双目拼接图建图

```python
import cv2
from src import StereoSLAM

slam = StereoSLAM(
    baseline=0.065,
    image_width=1280,
    image_height=720,
    stereo_width=2560,
    fov_horizontal=100.0,
    eye_order="right-left",
    auto_save_path=".runtime/live_map.json",
)

frame = cv2.imread("frame.png")  # shape: (720, 2560, 3)
result = slam.process_stereo_image(frame)

print(result["success"])
print(result["total_map_points"])
print(result["camera_pose"])
```

### 3.2 批量输入多张图

```python
import cv2
from src import StereoSLAM

slam = StereoSLAM(eye_order="right-left")
frames = [cv2.imread(path) for path in image_paths]
results = slam.process_images(frames, save_map_path=".runtime/batch_map.json")
```

### 3.3 保存和读取地图

```python
from src import StereoSLAM

slam = StereoSLAM(eye_order="right-left")
slam.process_stereo_image(frame)
slam.save_map(".runtime/map.json")

loaded = StereoSLAM(eye_order="right-left")
loaded.load_map(".runtime/map.json")
print(loaded.get_map_statistics())
```

### 3.4 用已有地图定位

```python
import cv2
from src import StereoMapLocalizer

localizer = StereoMapLocalizer(
    map_path=".runtime/web_live_map.json",
    baseline=0.065,
    image_width=1280,
    image_height=720,
    stereo_width=2560,
    fov_horizontal=100.0,
    eye_order="right-left",
)

frame = cv2.imread("current_frame.png")
result = localizer.localize_stereo_image(frame)

print(result["success"])
print(result["quality"])
print(result["camera_position"])
print(result["num_pnp_inliers"])
```

### 3.5 运行建图 Web 测试

```powershell
cd D:\app\trackerProject\stereo_slam
.\.venv\Scripts\python.exe .\simple_web_slam.py
```

打开：

```text
http://localhost:9704
```

默认保存地图：

```text
.runtime/web_live_map.json
```

### 3.6 运行只读定位 Web 测试

```powershell
cd D:\app\trackerProject\stereo_slam
.\.venv\Scripts\python.exe .\map_localization_web.py --map-path .runtime\web_live_map.json
```

打开：

```text
http://localhost:9705
```

## 4. 数据格式和坐标约定

### 4.1 输入图像

`StereoSLAM` 和 `StereoMapLocalizer` 都支持以下输入：

| 输入类型 | 示例 | 说明 |
|---|---|---|
| `np.ndarray` | `frame` | OpenCV BGR 或灰度图 |
| 图片路径 | `"frame.png"` | 内部使用 `cv2.imread` 读取 |
| 多张图片 iterable | `[frame1, frame2]` | 仅建图 API 的批处理使用 |
| 4D batch | `np.stack(frames)` | shape 类似 `(N, 720, 2560, 3)` |

默认拼接图尺寸必须是 `2560x720`。如果你的摄像头输出 `2560x800`，Web 和 camera debug 工具会居中裁剪到 `2560x720`。

### 4.2 `eye_order`

| 值 | 含义 |
|---|---|
| `"left-right"` | 输入图左半边是左眼，右半边是右眼 |
| `"right-left"` | 输入图左半边是右眼，右半边是左眼 |

当前 device 0 实测通常应使用：

```python
eye_order="right-left"
```

### 4.3 位姿和坐标

| 名称 | 说明 |
|---|---|
| `camera_pose` | world-to-camera 的 `4x4` 齐次变换矩阵 |
| `camera_position` | 相机中心在地图世界坐标中的位置 `[x, y, z]` |
| 地图点坐标 | 世界坐标，单位米 |
| 三角化点 | 先在相机坐标中计算，再转换为世界坐标写入地图 |

## 5. 推荐公开 API

推荐业务代码优先使用这些入口：

```python
from src import StereoSLAM, StereoMapLocalizer
```

如果导入环境不在项目根目录，也可以使用明确路径：

```python
from src.core.stereo_slam import StereoSLAM
from src.core.localization import StereoMapLocalizer
```

## 6. `StereoSLAM` 建图 API

### 6.1 构造函数

```python
StereoSLAM(
    device_id=0,
    baseline=0.065,
    focal_length=None,
    image_width=1280,
    image_height=720,
    stereo_width=None,
    fov_horizontal=100.0,
    eye_order="left-right",
    auto_save_path=None,
    debug_mode=False,
    logger=None,
)
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `device_id` | `int` | `0` | 设备 ID，只写入配置和地图统计，不会自动打开摄像头 |
| `baseline` | `float` | `0.065` | 双目基线，单位米 |
| `focal_length` | `float | None` | `None` | 焦距，单位像素。为 `None` 时用 `image_width` 和 `fov_horizontal` 推导 |
| `image_width` | `int` | `1280` | 单眼图像宽度 |
| `image_height` | `int` | `720` | 单眼图像高度 |
| `stereo_width` | `int | None` | `None` | 拼接图总宽度。为 `None` 时使用 `image_width * 2` |
| `fov_horizontal` | `float` | `100.0` | 单眼水平 FOV，单位度 |
| `eye_order` | `str` | `"left-right"` | 输入拼接顺序，只能是 `"left-right"` 或 `"right-left"` |
| `auto_save_path` | `str | Path | None` | `None` | 自动保存地图路径。每次 `process_stereo_image` 后保存 |
| `debug_mode` | `bool` | `False` | 是否在结果里附加 debug 信息 |
| `logger` | `logging.Logger | None` | `None` | 自定义日志对象 |

### 6.2 `split_stereo_image`

```python
left_image, right_image = slam.split_stereo_image(stereo_image)
```

| 参数 | 类型 | 说明 |
|---|---|---|
| `stereo_image` | `str | Path | np.ndarray` | 横向拼接双目图 |

返回：

| 返回值 | 类型 | 说明 |
|---|---|---|
| `left_image` | `np.ndarray` | 左眼图像 |
| `right_image` | `np.ndarray` | 右眼图像 |

### 6.3 `process_stereo_image`

```python
result = slam.process_stereo_image(stereo_image, frame_id=None, save_map_path=None)
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `stereo_image` | `str | Path | np.ndarray` | 必填 | 横向拼接双目图 |
| `frame_id` | `int | None` | `None` | 手动指定帧号。为 `None` 时使用地图内部计数器 |
| `save_map_path` | `str | Path | None` | `None` | 本次处理后保存地图到该路径。优先级高于 `auto_save_path` |

作用：

- 拆分左右眼图像
- 提取 ORB 特征
- 做左右眼匹配
- 三角化当前观测
- 估计视觉里程计
- 更新地图点
- 必要时添加关键帧
- 可选保存地图

### 6.4 `process_images`

```python
results = slam.process_images(images, save_map_path=None)
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `images` | `Iterable[str | Path | np.ndarray]` | 必填 | 多张横向拼接双目图 |
| `save_map_path` | `str | Path | None` | `None` | 批处理完成后保存地图 |

返回：`list[dict]`，每张图一个处理结果。

### 6.5 `process`

```python
result_or_results = slam.process(images, save_map_path=None)
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `images` | `str | Path | np.ndarray | Iterable` | 必填 | 单张图、多张图或 4D numpy batch |
| `save_map_path` | `str | Path | None` | `None` | 可选保存地图 |

返回规则：

| 输入 | 返回 |
|---|---|
| 单张图或单个路径 | `dict` |
| list/tuple/iterable | `list[dict]` |
| 4D numpy batch | `list[dict]` |

### 6.6 `process_frame`

```python
result = slam.process_frame(left_image, right_image, frame_id=None)
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `left_image` | `np.ndarray` | 必填 | 已拆分的左眼图 |
| `right_image` | `np.ndarray` | 必填 | 已拆分的右眼图 |
| `frame_id` | `int | None` | `None` | 手动帧号 |

适用于你已经自己完成左右眼拆分的场景。

### 6.7 地图和位姿方法

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `save_map(filepath)` | `filepath: str | Path` | `None` | 保存 JSON 地图 |
| `load_map(filepath)` | `filepath: str | Path` | `None` | 读取 JSON 地图 |
| `get_map_statistics()` | 无 | `dict` | 获取地图点数、边界、中心等统计 |
| `visualize_map(save_path=None)` | `save_path: str | Path | None` | `np.ndarray | None` | 生成简单俯视地图图像，可选保存 |
| `get_camera_pose()` | 无 | `np.ndarray` | 当前 world-to-camera 位姿矩阵 |
| `get_camera_position()` | 无 | `np.ndarray` | 当前相机中心世界坐标 |
| `get_last_observations()` | 无 | `list[dict]` | 当前帧三角化观测，主要用于 Web overlay |

### 6.8 `StereoSLAM` 结果字段

`process_stereo_image`、`process_frame` 等返回一个 `dict`。常见字段如下：

| 字段 | 类型 | 说明 |
|---|---|---|
| `success` | `bool` | 当前帧是否成功处理 |
| `error` | `str` | 失败原因。成功时通常不存在 |
| `frame_id` | `int` | 帧号 |
| `num_keypoints_left` | `int` | 左眼 ORB 特征点数 |
| `num_keypoints_right` | `int` | 右眼 ORB 特征点数 |
| `num_matches` | `int` | 双目匹配数 |
| `num_triangulated_points` | `int` | 通过质量检查的三角化点数 |
| `num_current_observations` | `int` | 当前帧可用于 overlay 的观测点数 |
| `num_new_points` | `int` | 新增地图点数 |
| `num_updated_points` | `int` | 更新地图点数 |
| `total_map_points` | `int` | 当前地图点总数 |
| `camera_pose` | `list[list[float]]` | world-to-camera 位姿矩阵 |
| `vo_matches` | `int` | 视觉里程计匹配数 |
| `vo_inliers` | `int` | PnP RANSAC 内点数 |
| `camera_moved_significant` | `bool` | 是否移动到需要添加关键帧 |
| `camera_movement_distance` | `float` | 相机移动距离，单位米 |
| `input_shape` | `list[int]` | 输入拼接图 shape |
| `left_shape` | `list[int]` | 左眼图 shape |
| `right_shape` | `list[int]` | 右眼图 shape |
| `eye_order` | `str` | 本次使用的左右眼顺序 |

## 7. `StereoMapLocalizer` 只读定位 API

### 7.1 构造函数

```python
StereoMapLocalizer(
    map_path=None,
    device_id=0,
    baseline=0.065,
    focal_length=None,
    image_width=1280,
    image_height=720,
    stereo_width=None,
    fov_horizontal=100.0,
    eye_order="left-right",
    ratio_threshold=0.75,
    min_pnp_matches=6,
    min_pnp_inliers=8,
    strong_pose_min_inliers=25,
    min_inliers_ratio=0.04,
    ransac_reproj_threshold=4.0,
    pnp_iterations=1000,
    max_descriptor_distance=72.0,
    max_pnp_matches=600,
    require_reciprocal_match=True,
    use_stereo_filter=False,
)
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `map_path` | `str | Path | None` | `None` | 要读取的地图 JSON。为 `None` 时可之后调用 `load_map` |
| `device_id` | `int` | `0` | 设备 ID，用于统计 |
| `baseline` | `float` | `0.065` | 双目基线，单位米 |
| `focal_length` | `float | None` | `None` | 焦距。为 `None` 时用 FOV 推导 |
| `image_width` | `int` | `1280` | 单眼宽度 |
| `image_height` | `int` | `720` | 单眼高度 |
| `stereo_width` | `int | None` | `None` | 拼接图总宽度 |
| `fov_horizontal` | `float` | `100.0` | 单眼水平 FOV |
| `eye_order` | `str` | `"left-right"` | 拼接顺序 |
| `ratio_threshold` | `float` | `0.75` | Lowe ratio test 阈值 |
| `min_pnp_matches` | `int` | `6` | PnP 前最少需要多少地图匹配 |
| `min_pnp_inliers` | `int` | `8` | 接受定位所需最少 PnP 内点 |
| `strong_pose_min_inliers` | `int` | `25` | 达到该内点数时，即使整体内点率较低也接受 |
| `min_inliers_ratio` | `float` | `0.04` | 接受定位所需最小内点率 |
| `ransac_reproj_threshold` | `float` | `4.0` | PnP RANSAC 重投影误差阈值，像素 |
| `pnp_iterations` | `int` | `1000` | PnP RANSAC 迭代次数 |
| `max_descriptor_distance` | `float` | `72.0` | ORB Hamming 距离上限 |
| `max_pnp_matches` | `int` | `600` | 最多送入 PnP 的匹配数，按 descriptor 距离排序 |
| `require_reciprocal_match` | `bool` | `True` | 是否要求 current-to-map 和 map-to-current 互为最近邻 |
| `use_stereo_filter` | `bool` | `False` | 是否只使用当前帧中存在左右眼匹配的左眼特征 |

### 7.2 方法

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `load_map(filepath)` | `filepath: str | Path` | `dict` | 读取地图并建立 descriptor 索引 |
| `get_map_statistics()` | 无 | `dict` | 返回地图统计和 `num_described_points` |
| `split_stereo_image(stereo_image)` | `str | Path | np.ndarray` | `(left, right)` | 拆分拼接图 |
| `localize_stereo_image(stereo_image, frame_id=None)` | 拼接图，可选帧号 | `dict` | 对拼接图定位，不更新地图 |
| `localize_frame(left_image, right_image, frame_id=None)` | 左眼图、右眼图、可选帧号 | `dict` | 对已拆分图定位 |
| `get_camera_position()` | 无 | `np.ndarray` | 最近一次成功定位的相机位置 |
| `project_map_points(camera_pose=None, image_shape=(720, 1280), max_points=1200)` | 可选位姿、图像 shape、最大点数 | `list[dict]` | 把地图点投影到当前图像平面 |

### 7.3 定位结果字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `success` | `bool` | 是否成功定位 |
| `quality` | `str` | `"ratio"`、`"strong"` 或 `"not_localized"` |
| `error` | `str | None` | 失败原因 |
| `num_map_points` | `int` | 地图点总数 |
| `num_described_map_points` | `int` | 有 ORB descriptor 的地图点数 |
| `num_keypoints_left` | `int` | 当前左眼 ORB 特征数 |
| `num_keypoints_right` | `int` | 当前右眼 ORB 特征数 |
| `num_stereo_matches` | `int` | 当前左右眼双目匹配数 |
| `num_candidate_features` | `int` | 用于地图匹配的当前特征数 |
| `num_raw_descriptor_matches` | `int` | 原始 descriptor 匹配数 |
| `num_ratio_matches` | `int` | 通过 Lowe ratio 的匹配数 |
| `num_distance_matches` | `int` | 通过 Hamming 距离上限的匹配数 |
| `num_reciprocal_matches` | `int` | 通过互为最近邻过滤的匹配数 |
| `num_map_matches` | `int` | 最终地图匹配数 |
| `num_pnp_used_matches` | `int` | 实际送入 PnP 的匹配数 |
| `num_pnp_inliers` | `int` | PnP RANSAC 内点数 |
| `inlier_ratio` | `float` | `num_pnp_inliers / num_pnp_used_matches` |
| `mean_inlier_reprojection_error` | `float | None` | PnP 内点平均重投影误差 |
| `median_inlier_reprojection_error` | `float | None` | PnP 内点中位重投影误差 |
| `descriptor_distance_median` | `float | None` | descriptor 距离中位数 |
| `camera_pose` | `list[list[float]]` | 成功定位后的 world-to-camera 矩阵 |
| `camera_position` | `list[float]` | 成功定位后的相机世界坐标 |
| `candidate_camera_pose` | `list[list[float]] | None` | 即使未接受也会返回的候选位姿 |
| `candidate_camera_position` | `list[float] | None` | 候选相机位置 |
| `matched_map_points` | `list[dict]` | 匹配地图点、当前特征点、投影点、误差等 |
| `visible_map_points` | `list[dict]` | 当前位姿下可见的地图点投影 |

## 8. 配置类

这些 dataclass 主要用于集中管理参数。通常业务代码不需要直接创建它们，除非你要扩展内部实现。

### 8.1 `CameraConfig`

```python
CameraConfig(
    image_width=1280,
    image_height=720,
    fov_horizontal=100.0,
    baseline=0.065,
    focal_length=1000.0,
    principal_point=None,
)
```

| 参数 | 说明 |
|---|---|
| `image_width` | 单眼宽度 |
| `image_height` | 单眼高度 |
| `fov_horizontal` | 单眼水平 FOV |
| `baseline` | 双目基线，单位米 |
| `focal_length` | 焦距，单位像素 |
| `principal_point` | 主点 `(cx, cy)`。为 `None` 时使用图像中心 |

方法：

| 方法 | 返回 | 说明 |
|---|---|---|
| `get_intrinsics()` | `np.ndarray` | 返回 `3x3` 相机内参矩阵 |

### 8.2 `FeatureConfig`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `n_features` | `2000` | ORB 最大特征数 |
| `n_levels` | `8` | ORB 金字塔层数 |
| `edge_threshold` | `31` | ORB 边缘阈值 |
| `first_level` | `0` | 金字塔起始层 |
| `scale_factor` | `1.2` | 金字塔缩放因子 |

### 8.3 `MatchingConfig`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `ratio_threshold` | `0.75` | Lowe ratio test 阈值 |
| `cross_check` | `False` | 是否使用 BFMatcher cross-check |
| `max_vertical_diff` | `20.0` | 双目匹配最大 y 偏差 |

### 8.4 `VOConfig`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `min_inliers_ratio` | `0.15` | VO PnP 最小内点率 |
| `ransac_reproj_threshold` | `3.0` | VO PnP 重投影误差阈值 |
| `min_matches` | `10` | VO 最小匹配数 |

### 8.5 `MapConfig`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `distance_threshold` | `0.05` | 地图点空间关联距离阈值，米 |
| `max_observation_distance` | `0.15` | 最大观测距离阈值，米 |
| `min_observations` | `3` | 地图点保留所需最少观测次数 |
| `min_disparity` | `2.0` | 最小视差，像素 |
| `max_disparity` | `300.0` | 最大视差，像素 |
| `max_vertical_disparity` | `20.0` | 左右眼匹配最大 y 偏差 |
| `min_depth` | `0.25` | 最小深度，米 |
| `max_depth` | `12.0` | 最大深度，米 |
| `max_cache_size` | `500` | VO 缓存最大点数 |
| `update_weight` | `0.05` | 更新已有地图点位置时的新观测权重 |
| `depth_variance_threshold` | `0.05` | 深度稳定性阈值 |
| `min_stereo_baseline` | `0.02` | 最小双目基线变化 |
| `min_reprojection_error` | `0.5` | 重投影误差配置项 |
| `min_parallax_angle` | `5.0` | 最小视差角 |
| `min_keyframe_distance` | `0.1` | 添加关键帧所需最小移动距离，米 |
| `min_keyframe_angle` | `10.0` | 添加关键帧所需最小旋转角度 |
| `max_reprojection_pixel_error` | `3.0` | 2D 投影关联搜索半径 |

### 8.6 `SLAMConfig`

```python
SLAMConfig(device_id=0, debug_mode=False)
```

| 参数 | 说明 |
|---|---|
| `device_id` | 设备 ID |
| `debug_mode` | 是否启用 debug |
| `camera` | `CameraConfig` |
| `feature` | `FeatureConfig` |
| `matching` | `MatchingConfig` |
| `vo` | `VOConfig` |
| `map` | `MapConfig` |

方法：

| 方法 | 返回 | 说明 |
|---|---|---|
| `get_focal_length_from_fov()` | `float` | 根据 FOV 和图像宽度计算焦距 |

## 9. 特征处理 API

### 9.1 `FeatureExtractor`

```python
FeatureExtractor(
    n_features=2000,
    n_levels=8,
    edge_threshold=31,
    first_level=0,
    scale_factor=1.2,
)
```

| 参数 | 说明 |
|---|---|
| `n_features` | ORB 最大特征点数 |
| `n_levels` | ORB 金字塔层数 |
| `edge_threshold` | ORB 边缘阈值 |
| `first_level` | ORB 起始层 |
| `scale_factor` | ORB 金字塔缩放因子 |

方法：

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `extract(image)` | `image: np.ndarray` | `(keypoints, descriptors)` | 从单张图提取 ORB |
| `extract_stereo(left_img, right_img)` | 左右眼图 | `(left_keypoints, right_keypoints, left_descriptors, right_descriptors)` | 分别提取左右眼 ORB |

### 9.2 `StereoMatcher`

```python
StereoMatcher(ratio_threshold=0.75, cross_check=False)
```

| 参数 | 说明 |
|---|---|
| `ratio_threshold` | Lowe ratio test 阈值 |
| `cross_check` | 是否使用 cross-check |

方法：

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `match_stereo_features(left_keypoints, right_keypoints, left_descriptors, right_descriptors)` | 左右眼 keypoints 和 descriptors | `list[(DMatch, DMatch)]` | KNN 匹配并做 ratio test |
| `match_stereo_rectified(left_keypoints, right_keypoints, left_descriptors, right_descriptors, max_vertical_diff=2.0, min_disparity=1.0, max_disparity=1000.0)` | 左右眼特征和过滤阈值 | `list[DMatch]` | 适合已近似校正的双目图，额外过滤 y 偏差和视差 |

### 9.3 `FeatureTracker`

```python
FeatureTracker(max_features=1000, quality_level=0.01, min_distance=10)
```

| 参数 | 说明 |
|---|---|
| `max_features` | 最大角点数 |
| `quality_level` | `cv2.goodFeaturesToTrack` 质量阈值 |
| `min_distance` | 角点之间的最小距离 |

方法：

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `detect_features(gray)` | 灰度图 | `(keypoints, mask)` | 检测角点并转换为 `cv2.KeyPoint` |
| `track_features(prev_gray, curr_gray, prev_keypoints)` | 前一帧灰度图、当前灰度图、前一帧 keypoints | `(prev_pts, curr_pts, status)` | Lucas-Kanade 光流跟踪 |
| `update(gray)` | 当前灰度图 | `None` | 更新内部上一帧缓存 |

## 10. 几何 API

### 10.1 `StereoTriangulator`

```python
StereoTriangulator(
    baseline=0.065,
    focal_length=1000.0,
    principal_point=(1280.0, 360.0),
)
```

| 参数 | 说明 |
|---|---|
| `baseline` | 双目基线，米 |
| `focal_length` | 焦距，像素 |
| `principal_point` | 主点 `(cx, cy)` |

方法：

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `triangulate_point(left_point, right_point)` | 左右眼 2D 点 `(u, v)` | `np.ndarray | None` | 返回 `[x, y, z]`，负视差返回 `None` |
| `triangulate_matches(left_keypoints, right_keypoints, matches)` | 左右眼 keypoints 和 DMatch 列表 | `list[(feature_id, position)]` | 批量三角化匹配点 |

### 10.2 `PoseEstimator`

```python
PoseEstimator(
    K,
    distortion_coeffs=None,
    min_inliers_ratio=0.15,
    ransac_reproj_threshold=3.0,
)
```

| 参数 | 说明 |
|---|---|
| `K` | `3x3` 相机内参矩阵 |
| `distortion_coeffs` | 畸变参数。为 `None` 时使用零畸变 |
| `min_inliers_ratio` | 最小内点率 |
| `ransac_reproj_threshold` | RANSAC 重投影误差阈值 |

方法：

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `estimate_pose(object_points, image_points)` | 3D 点 `(N,3)` 和 2D 点 `(N,2)` | `(pose, num_inliers)` | 用 PnP RANSAC 估计位姿 |
| `get_pose()` | 无 | `np.ndarray` | 返回当前位姿 |
| `reset()` | 无 | `None` | 重置位姿 |

### 10.3 `GeometryUtils`

这些是静态工具函数，不需要创建对象。

| 函数 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `project_3d_to_2d_with_depth(points_3d, camera_pose, K, image_shape, return_indices=False)` | 3D 点、world-to-camera 位姿、内参、图像 shape、是否返回索引 | `np.ndarray` 或 `(projected, indices)` | 投影 3D 点，保留相机深度 |
| `project_3d_to_2d(points_3d, camera_pose, K, image_shape)` | 3D 点、位姿、内参、图像 shape | `np.ndarray` | 投影到 2D，不返回深度 |
| `find_nearby_point(position, existing_points, threshold=0.05)` | 新点、已有点 dict、距离阈值 | `int | None` | 查找空间上最近的已有点 |
| `create_pose_matrix(R, t)` | 旋转矩阵和平移向量 | `np.ndarray` | 创建 `4x4` 位姿矩阵 |
| `pose_to_se3(pose)` | `4x4` 位姿 | `(R, t)` | 拆分旋转和平移 |

## 11. 地图 API

### 11.1 `Map`

```python
Map(device_id=0, min_observations=2)
```

| 参数 | 说明 |
|---|---|
| `device_id` | 设备 ID |
| `min_observations` | 地图点最少观测次数，低于该值可被剔除 |

方法：

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `add_3d_point(position, color=None, descriptor=None, observation_ids=None)` | 3D 坐标、颜色、ORB descriptor、观测帧 ID | `int` | 添加地图点，返回 point id |
| `update_3d_point(point_id, position=None, color=None, descriptor=None, add_observation=None, use_weighted_average=True, update_weight=0.3)` | 点 ID 和可选更新字段 | `None` | 更新地图点 |
| `remove_3d_point(point_id)` | 点 ID | `None` | 删除地图点 |
| `cull_insecure_points()` | 无 | `int` | 删除观测次数不足的点，返回删除数量 |
| `filter_points_by_depth(min_depth=0.5, max_depth=20.0)` | 深度范围 | `int` | 删除深度范围外的点 |
| `add_keyframe(frame_id, left_image, right_image, left_keypoints, right_keypoints, left_descriptors, right_descriptors, camera_pose=None)` | 关键帧数据 | `KeyFrame` | 添加关键帧 |
| `get_keyframe(frame_id)` | 帧 ID | `KeyFrame | None` | 获取指定关键帧 |
| `get_all_keyframes()` | 无 | `list[KeyFrame]` | 获取所有关键帧 |
| `get_3d_points_array()` | 无 | `np.ndarray` | 返回所有地图点坐标 |
| `get_3d_points_colors()` | 无 | `np.ndarray | None` | 返回所有地图点颜色 |
| `get_described_points()` | 无 | `(point_ids, positions, descriptors)` | 返回带 ORB descriptor 的地图点 |
| `get_statistics()` | 无 | `dict` | 地图统计 |
| `save_to_file(filepath)` | 文件路径 | `None` | 保存 JSON 地图 |
| `Map.load_from_file(filepath)` | 文件路径 | `Map` | 从 JSON 加载地图 |

### 11.2 `Point3D`

```python
Point3D(
    position,
    color=None,
    descriptor=None,
    observation_count=0,
    last_seen_frame=0,
    observation_ids=[],
)
```

| 参数 | 说明 |
|---|---|
| `position` | 3D 坐标 `[x, y, z]` |
| `color` | BGR 颜色，可选 |
| `descriptor` | ORB descriptor，长度通常是 32 |
| `observation_count` | 观测次数 |
| `last_seen_frame` | 最后观测帧号 |
| `observation_ids` | 观测该点的帧号列表 |

方法：

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `mark_observed(frame_id)` | 帧号 | `None` | 记录一次观测 |
| `add_observation(frame_id, position, weight=1.0, use_weighted_average=True)` | 帧号、新位置、权重、是否加权平均 | `None` | 记录观测并更新位置 |
| `update_descriptor(descriptor)` | ORB descriptor | `None` | 更新 descriptor |
| `get_confidence()` | 无 | `float` | 基于观测次数的置信度 |
| `should_cull(min_observations=2)` | 最少观测次数 | `bool` | 是否应被剔除 |

### 11.3 `KeyFrame`

```python
KeyFrame(
    frame_id,
    timestamp,
    left_image=None,
    right_image=None,
    left_keypoints=[],
    right_keypoints=[],
    left_descriptors=None,
    right_descriptors=None,
    camera_pose=None,
    matched_3d_points=[],
)
```

| 字段 | 说明 |
|---|---|
| `frame_id` | 帧号 |
| `timestamp` | 时间戳 |
| `left_image` / `right_image` | 左右眼图像 |
| `left_keypoints` / `right_keypoints` | 左右眼特征点 |
| `left_descriptors` / `right_descriptors` | 左右眼 ORB descriptor |
| `camera_pose` | 关键帧相机位姿 |
| `matched_3d_points` | 关联地图点 ID |

类方法：

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `KeyFrame.create(...)` | 与构造关键字段相同 | `KeyFrame` | 创建关键帧的工厂方法 |

## 12. 视觉里程计 API

### 12.1 `VisualOdometry`

```python
VisualOdometry(
    K,
    distortion_coeffs=None,
    min_inliers_ratio=0.15,
    ransac_reproj_threshold=3.0,
)
```

| 参数 | 说明 |
|---|---|
| `K` | 相机内参 |
| `distortion_coeffs` | 畸变参数 |
| `min_inliers_ratio` | 最小内点率 |
| `ransac_reproj_threshold` | PnP RANSAC 重投影误差阈值 |

方法：

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `initialize(keypoints, descriptors, three_d_points)` | 初始帧 keypoints、descriptor、对应 3D 点 | `None` | 初始化 VO |
| `update(keypoints, descriptors)` | 当前帧 keypoints 和 descriptor | `(camera_pose, num_matches, num_inliers)` | 更新位姿 |
| `get_pose()` | 无 | `np.ndarray` | 返回当前位姿 |
| `reset()` | 无 | `None` | 重置 VO |

### 12.2 `MapUpdater`

```python
MapUpdater(distance_threshold=0.1)
```

| 参数 | 说明 |
|---|---|
| `distance_threshold` | 判断新点和已有点是否匹配的空间距离阈值 |

方法：

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `find_matching_points(new_points, existing_points)` | 新点 `(N,3)`、已有点 `(M,3)` | `(new_indices, existing_indices)` | 查找空间近邻匹配 |
| `update_map_points(existing_points, new_observations, match_indices)` | 已有点、新观测、匹配索引 | `np.ndarray` | 用平均位置更新已有点 |

## 13. Web、摄像头和测试工具 API

### 13.1 摄像头工具

```python
from src.web.camera import open_stereo_camera, normalize_camera_frame
```

| 函数 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `open_stereo_camera(device_id=0, width=2560, height=720)` | 设备 ID、期望宽高 | `cv2.VideoCapture` | 打开摄像头并设置分辨率和 FPS |
| `normalize_camera_frame(frame, expected_width=2560, expected_height=720)` | 摄像头帧、期望宽高 | `np.ndarray | None` | 宽度匹配且高度过高时居中裁剪 |

### 13.2 合成测试图

```python
from src.web import create_synthetic_stereo_images, create_synthetic_stereo_frame
```

| 函数 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `create_synthetic_stereo_images(width=1280, height=720)` | 单眼宽高 | `(left_img, right_img)` | 创建一组合成双目图 |
| `create_synthetic_stereo_frame(width=2560, height=720, eye_order="left-right")` | 拼接图宽高、左右顺序 | `np.ndarray` | 创建一张横向拼接合成双目图 |

### 13.3 建图 Web

```python
from src.web import run_web_slam, create_app
```

| 函数 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `run_web_slam(host="0.0.0.0", port=9704, save_map_path=DEFAULT_WEB_MAP_PATH, save_every_n_frames=30)` | Flask host、端口、地图保存路径、保存频率 | `None` | 启动实时建图 Web |
| `create_app(state=None)` | 可选 `WebSLAMState` | `Flask` | 创建 Flask app，适合二次集成 |

`run_web_slam` 参数说明：

| 参数 | 说明 |
|---|---|
| `host` | Flask 监听地址 |
| `port` | Flask 端口 |
| `save_map_path` | 地图 JSON 保存路径。为 `None` 时不保存 |
| `save_every_n_frames` | 每多少帧保存一次。小于等于 0 时只在退出清理时保存 |

### 13.4 只读定位 Web

```python
from src.web import run_localization_web, create_localization_app
```

| 函数 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `run_localization_web(...)` | 见下表 | `None` | 启动只读定位 Web |
| `create_localization_app(state=None)` | 可选 `LocalizationWebState` | `Flask` | 创建定位 Flask app |

`run_localization_web` 参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `map_path` | `.runtime/web_live_map.json` | 读取的地图 |
| `host` | `"0.0.0.0"` | Flask host |
| `port` | `9705` | Flask 端口 |
| `device_id` | `0` | 摄像头 ID |
| `baseline` | `0.065` | 双目基线，米 |
| `width` | `2560` | 拼接图宽度 |
| `height` | `720` | 拼接图高度 |
| `fov_horizontal` | `100.0` | 单眼水平 FOV |
| `eye_order` | `"right-left"` | 左右眼顺序 |
| `use_stereo_filter` | `False` | 是否只用当前帧左右眼能匹配的特征 |
| `min_pnp_inliers` | `8` | 接受定位最少 PnP 内点 |
| `strong_pose_min_inliers` | `25` | strong pose 最少内点 |
| `min_inliers_ratio` | `0.04` | 最小内点率 |
| `max_descriptor_distance` | `72.0` | 最大 ORB Hamming 距离 |
| `max_pnp_matches` | `600` | 最多送入 PnP 的匹配数 |
| `require_reciprocal_match` | `True` | 是否要求互为最近邻匹配 |

## 14. 命令行脚本

### 14.1 `simple_web_slam.py`

用途：运行实时建图 Web，会更新地图。

```powershell
.\.venv\Scripts\python.exe .\simple_web_slam.py --save-map .runtime\web_live_map.json
```

参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--host` | `0.0.0.0` | Flask host |
| `--port` | `9704` | Flask 端口 |
| `--save-map` | `.runtime/web_live_map.json` | 自动保存地图路径 |
| `--save-every` | `30` | 每多少帧保存一次。`0` 表示只在退出时保存 |
| `--no-save-map` | `False` | 禁用保存 |

### 14.2 `map_localization_web.py`

用途：运行只读定位 Web，不更新地图。

```powershell
.\.venv\Scripts\python.exe .\map_localization_web.py --map-path .runtime\web_live_map.json
```

参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--host` | `0.0.0.0` | Flask host |
| `--port` | `9705` | Flask 端口 |
| `--device` | `0` | 摄像头 ID |
| `--width` | `2560` | 拼接图宽度 |
| `--height` | `720` | 拼接图高度 |
| `--baseline` | `0.065` | 双目基线，米 |
| `--fov` | `100.0` | 单眼水平 FOV |
| `--eye-order` | `right-left` | 左右眼顺序 |
| `--map-path` | `.runtime/web_live_map.json` | 要读取的地图 |
| `--stereo-filter` | `False` | 只用当前帧左右眼有匹配的特征 |
| `--min-inliers` | `8` | 接受定位最少 PnP 内点 |
| `--strong-inliers` | `25` | strong pose 内点阈值 |
| `--min-inlier-ratio` | `0.04` | 最小内点率 |
| `--max-descriptor-distance` | `72.0` | ORB Hamming 距离上限 |
| `--max-pnp-matches` | `600` | 最多送入 PnP 的匹配数 |
| `--no-reciprocal-match` | `False` | 禁用互为最近邻过滤 |

### 14.3 `scripts/camera_debug.py`

用途：从摄像头抓取固定帧数建图，保存地图，不启动 Web。

```powershell
.\.venv\Scripts\python.exe .\scripts\camera_debug.py --device 0 --frames 60 --eye-order right-left
```

参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--device` | `0` | 摄像头 ID |
| `--frames` | `60` | 处理多少帧 |
| `--width` | `2560` | 拼接图宽度 |
| `--height` | `720` | 处理高度 |
| `--baseline` | `0.065` | 双目基线，米 |
| `--fov` | `100.0` | 单眼水平 FOV |
| `--eye-order` | `right-left` | 左右眼顺序 |
| `--save-map` | `.runtime/device0_map.json` | 输出地图路径 |

## 15. 常见问题

### 15.1 页面显示黑底彩色点

这通常表示建图 Web 没有打开真实摄像头，退回到了合成帧。检查：

- 摄像头是否被其他进程占用
- device ID 是否正确
- 是否已有另一个 Web app 在使用 device 0

### 15.2 定位页面显示 `Loaded map has too few ORB descriptors`

说明地图是旧代码保存的，没有 `descriptor` 字段。需要重新运行建图：

```powershell
.\.venv\Scripts\python.exe .\simple_web_slam.py --save-map .runtime\web_live_map.json
```

### 15.3 有很多 `Map Matches` 但 `not localized`

看 `PnP Inliers`、`Inlier Ratio` 和 `Median Error`：

- `PnP Inliers` 太低：当前视角和地图差异大，或者地图质量差。
- `Median Error` 很大：匹配几何不一致，可能左右眼顺序错、地图漂移、FOV/基线不准。
- 可以临时放宽定位参数，例如 `--min-inliers 6`，但更好的做法是重新建质量更稳定的地图。

### 15.4 地图点很多但定位不稳定

当前库没有 bundle adjustment 和回环优化。地图可能因为视觉里程计漂移而不一致。建议：

- 尽量从稳定视角慢速建图
- 避免大面积纯白墙、反光电视屏、强曝光区域
- 使用固定的 `eye_order`
- 后续加入双目标定和畸变矫正

## 16. 推荐使用顺序

新程序员建议按这个顺序上手：

1. 跑测试，确认环境正确。
2. 运行 `simple_web_slam.py`，确认能看到真实摄像头画面。
3. 移动摄像头建图，保存 `.runtime/web_live_map.json`。
4. 运行 `map_localization_web.py`，确认能读取地图定位。
5. 在业务代码里使用 `StereoSLAM` 建图。
6. 在业务代码里使用 `StereoMapLocalizer` 只读定位。

