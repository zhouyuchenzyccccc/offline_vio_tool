# 3D 实时可视化指南

## 功能

集成 Open3D 的 3D 可视化，支持：
- **第三视角观看**：交互式相机控制
- **相机位姿可视化**：彩色金字体显示相机视锥
  - 🔵 蓝色：普通帧
  - 🔴 红色：关键帧（需要后续实现）
- **实时点云**：从深度图投影生成
- **轨迹连接线**：绿色线连接所有相机位置
- **坐标系框架**：RGB 坐标轴（红=X，绿=Y，蓝=Z）

## 安装依赖

```bash
# 安装 Open3D 和 scipy（用于四元数转换）
pip install open3d scipy
```

## 使用方法

### 1️⃣ 基础可视化（仅轨迹和相机）

```bash
python run_vio.py \
  --dataset /path/to/dataset \
  --cam_id 07 \
  --sensor_mode rgbd \
  --run_slam \
  --orb_exec ./build/rgbd_inertial_offline \
  --vocab /path/to/ORBvoc.txt \
  --settings cam07_settings.yaml \
  --out_dir ./output_cam07 \
  --visualize
```

**交互操作：**
- **鼠标左键 + 拖动**：旋转视图
- **鼠标右键 + 拖动**：平移视图
- **鼠标滚轮**：缩放
- **Q**：退出

### 2️⃣ 带点云的完整可视化（较慢）

```bash
python run_vio.py \
  --dataset /path/to/dataset \
  --cam_id 07 \
  --sensor_mode rgbd \
  --run_slam \
  --orb_exec ./build/rgbd_inertial_offline \
  --vocab /path/to/ORBvoc.txt \
  --settings cam07_settings.yaml \
  --out_dir ./output_cam07 \
  --visualize \
  --vis_with_pointcloud
```

> **注意**：构建点云需要加载和处理所有图像，可能较慢。使用 `--max_frames 100` 限制帧数。

### 3️⃣ 快速演示（前 50 帧 + 点云）

```bash
python run_vio.py \
  --dataset /path/to/dataset \
  --cam_id 07 \
  --sensor_mode rgbd \
  --run_slam \
  --orb_exec ./build/rgbd_inertial_offline \
  --vocab /path/to/ORBvoc.txt \
  --settings cam07_settings.yaml \
  --out_dir ./output_demo \
  --max_frames 50 \
  --visualize \
  --vis_with_pointcloud
```

## 可视化要素说明

| 要素 | 含义 | 颜色 |
|------|------|------|
| 绿色线 | 相机轨迹 | Green |
| 蓝色金字体 | 相机视锥（普通帧） | Blue |
| 红色金字体 | 相机视锥（关键帧） | Red |
| RGB 坐标轴 | 世界坐标系原点 | R/G/B |
| 彩色点云 | RGB-D 投影的 3D 点 | RGB |

## 控制提示

### 视图调整

```python
# 在代码中修改默认视角（vio_tool/realtime_visualizer.py）
self.view_control.set_zoom(0.8)      # 缩放级别
self.view_control.rotate(100, 50)    # 旋转角度 (horizontal, vertical)
```

### 性能优化

- **点云采样率**：减少处理的像素数
  ```python
  sample_rate=5  # 每 5 个像素采样 1 个（详见代码）
  ```

- **最大深度**：过滤远距离或近距离的点
  - 依赖于 `depth > 0.1` 检查（见 `realtime_visualizer.py`）

## 常见问题

### Q: "ModuleNotFoundError: No module named 'open3d'"

**A:** 安装 Open3D：
```bash
pip install open3d
```

### Q: 点云变成垃圾或不可见

**A:** 检查以下几点：
1. 深度图是否正确（应为 `uint16` mm 或类似）
2. `--depth_scale` 是否正确（Orbbec 默认 1000）
3. 相机内参是否正确（检查 YAML 文件）
4. 相机位姿是否有效（检查轨迹是否连贯）

### Q: 运行缓慢或吃内存

**A:** 降低点云采样率或限制帧数：
```bash
--max_frames 100 --visualize --vis_with_pointcloud
```

### Q: 如何保存截图

**A:** 在交互窗口中按 `Ctrl+C` 后修改代码：
```python
visualizer.save_screenshot("output.png")
```

## 进阶：自定义可视化

编辑 `vio_tool/realtime_visualizer.py` 的 `CameraPoseVisualizer` 类：

```python
# 修改相机大小
visualizer.add_camera_model(pose, size=0.05)

# 修改点大小
visualizer.render_options.point_size = 2.0

# 修改背景颜色
visualizer.render_options.background_color = np.array([0.8, 0.8, 0.8])
```

## 相关文件

- **主脚本**：`run_vio.py`
- **可视化模块**：`vio_tool/realtime_visualizer.py`
- **配置文件**：`cam07_settings.yaml`
