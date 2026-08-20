# Supermarket Sorting Task

> DG-202606 V2.0 note: official-style testing now uses the split images
> `supermarket_sorting:server` and `supermarket_sorting:client`, mounts this
> repository at `/workspace/baseline`, and reads tasks from
> `/supermarket_sorting/task`. See `RUN_V2_OFFICIAL.md` first. Older commands
> below are kept only for legacy/local reference.

超市分拣比赛 ROS2 仿真环境。server 运行 MuJoCo + 可选 3DGS 场景，发布相机、深度、里程计和关节状态；参赛 client 通过 ROS2 话题控制 MMK2 机器人完成货架取货和配送台放置。

## 部署

宿主机需要 Docker、NVIDIA Driver、NVIDIA Container Toolkit 和 NVIDIA GPU。拉取比赛镜像：

```bash
docker pull crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/supermarket_sorting:latest
```

如需从源码重新构建镜像：

```bash
docker build -t crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/supermarket_sorting:latest .
```

允许运行窗口显示，并创建运行缓存：

```bash
xhost +local:docker
docker volume create supermarket_sorting_cache
```

## 固定 Baseline

这一组命令用于验证 baseline。场景使用固定随机种子：baseline 面对的货架只有中间第二排第二个位置是可乐，baseline 会通过视觉检测抓取该可乐并完成一次搬运。

### Server

```bash
docker run --rm -it \
  --gpus all \
  --network host \
  --ipc host \
  --name supermarket_sorting_server \
  -e DISPLAY=${DISPLAY} \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e MUJOCO_GL=glfw \
  -e SUPERMARKET_HEADLESS=0 \
  -e SUPERMARKET_ENABLE_RENDER=1 \
  -e SUPERMARKET_USE_GS=1 \
  -e SUPERMARKET_RANDOMIZE=1 \
  -e SUPERMARKET_SEED=11 \
  -e SUPERMARKET_ENABLE_SCORE=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v supermarket_sorting_cache:/root/.cache \
  crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/supermarket_sorting:latest \
  bash -lc "cd /workspace/supermarket_sorting_task && source /opt/ros/humble/setup.bash && python3 examples/supermarket_sorting/supermarket_sorting_server.py"
```

### Baseline Client

baseline client 同时启动可乐视觉检测节点。抓取目标来自 `/kele/detections` 的视觉世界坐标；检测不到时等待，不回退到固定槽位坐标。

```bash
docker run --rm -it \
  --gpus all \
  --network host \
  --ipc host \
  --name supermarket_sorting_baseline \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -v supermarket_sorting_cache:/root/.cache \
  crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/supermarket_sorting:latest \
  bash -lc 'cd /workspace/supermarket_sorting_task
    source /opt/ros/humble/setup.bash
    python3 examples/supermarket_sorting/perception/kele_detect.py --backend yolo &
    det_pid=$!
    trap "kill $det_pid 2>/dev/null || true" EXIT
    python3 examples/supermarket_sorting/supermarket_sorting_client.py'
```

## 正式 Server

正式评测使用随机布局并开启裁判。每次启动 45 个商品都会重新随机打乱；不固定抓取目标，任意商品完成有效搬运都可以得分。

```bash
docker run --rm -it \
  --gpus all \
  --network host \
  --ipc host \
  --name supermarket_sorting_server \
  -e DISPLAY=${DISPLAY} \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e MUJOCO_GL=glfw \
  -e SUPERMARKET_HEADLESS=0 \
  -e SUPERMARKET_ENABLE_RENDER=1 \
  -e SUPERMARKET_USE_GS=1 \
  -e SUPERMARKET_RANDOMIZE=1 \
  -e SUPERMARKET_ENABLE_SCORE=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v supermarket_sorting_cache:/root/.cache \
  crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/supermarket_sorting:latest \
  bash -lc "cd /workspace/supermarket_sorting_task && source /opt/ros/humble/setup.bash && python3 examples/supermarket_sorting/supermarket_sorting_server.py"
```

## 评分标准

比赛时间为 10 分钟（600 秒，与 `referee_json/retail_referee_config.json` 的 `time_limit_s` 一致；旧版规则为 7 分钟）。计分单位是一次完整搬运流程：从货架取出一个商品，搬到配送区并完成放置。任意商品都可以得分，每个商品最多计分一次。

| 步骤 | 分值 | 判定含义 |
| --- | ---: | --- |
| S1 | +1 | 机器人到达取货区 |
| S2 | +2 | 夹爪够到目标物体 |
| S3 | +6 | 成功抓取并撤出货架 |
| S4 | +1 | 携带目标到达配送区 |
| S5 | +10 | 目标准确放入配送框 |
| 直立稳定 | +5 | 放置后静止且姿态直立 |

单个商品满分 25 分。流程中发生结构碰撞扣 5 分，碰倒其他商品扣 5 分。已交付/已掉落的商品不再参与"碰倒"判定；掉落使本轮作废，且按官方 V2 规则"本轮失败清除本轮扣分"，本轮成绩为 0（扣分只作用于最终成功的循环）。

> **官方 V2 评分对照**（本仓库裁判是开发预言机，正式成绩以官方自动裁判为准）：
> 任务完成分 80（每个"取货+送货"循环 12 分 × 5 订单）+ 流程效率奖励 20
> （自主优化取送顺序 10 + 无长时间停顿连续作业 10）+ 技术评价 20
> （动态避障与重规划 10 + 视觉伺服与抓取放置策略 10）。扣分项：抓取中掉落 -4、
> 移动中掉落 -3、定位不准碰到邻位物品每次 -1（同一物品最多 -2）、每次人工干预 -5；
> 本轮失败则成绩为 0；危险行为裁判可终止比赛。客户端已支持：任意顺序执行订单
> （`SUPERMARKET_LOCK_OFFICIAL_ORDER` 默认关闭）、库存驱动的目标代价排序、抓取前
> 重新视觉确认、ArUco 货位关联与 RGB-D 最终位姿监控。

## ROS2 话题

`ROS_DOMAIN_ID` 必须在 server 和 client 之间保持一致。下表只列比赛相关接口，不包含 `/rosout`、`/parameter_events` 等 ROS2 自动话题。

### Server 发布

| Topic | Type | 说明 |
| --- | --- | --- |
| `/slamware_ros_sdk_server_node/odom` | `nav_msgs/msg/Odometry` | 底盘世界位姿和速度 |
| `/tf` | `tf2_msgs/msg/TFMessage` | 动态 TF，主要为 `odom -> base_link` |
| `/joint_states` | `sensor_msgs/msg/JointState` | 升降、头部、双臂和夹爪关节状态 |
| `/head_camera/color/image_raw` | `sensor_msgs/msg/Image` | 头部 RGB 图像，默认 640x480 |
| `/head_camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | 头部 RGB 相机内参 |
| `/head_camera/aligned_depth_to_color/image_raw` | `sensor_msgs/msg/Image` | 头部深度图，`mono16`，单位毫米 |
| `/head_camera/aligned_depth_to_color/camera_info` | `sensor_msgs/msg/CameraInfo` | 头部深度相机内参 |
| `/left_camera/color/image_raw` | `sensor_msgs/msg/Image` | 左腕 RGB 图像 |
| `/left_camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | 左腕相机内参 |
| `/right_camera/color/image_raw` | `sensor_msgs/msg/Image` | 右腕 RGB 图像 |
| `/right_camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | 右腕相机内参 |
| `/referee/taskinfo` | `std_msgs/msg/String` | 裁判目标清单 |
| `/referee/gameinfo` | `std_msgs/msg/String` | 实时分数、完成数和流程步号 |
| `/referee/score` | `std_msgs/msg/Int32` | 当前总分 |

### Server 订阅

| Topic | Type | 控制格式 |
| --- | --- | --- |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | `linear.x` 前进速度，`angular.z` 角速度 |
| `/spine_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | `[slide_joint]` |
| `/head_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | `[head_yaw_joint, head_pitch_joint]` |
| `/left_arm_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | `[joint1, joint2, joint3, joint4, joint5, joint6, gripper]` |
| `/right_arm_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | `[joint1, joint2, joint3, joint4, joint5, joint6, gripper]` |

### 裁判服务

| Service | Type | 说明 |
| --- | --- | --- |
| `/supermarket_sorting/reset_run` | `std_srvs/srv/Trigger` | 不重启 server 开始新一局；会重置仿真状态、分数和目标快照 |

### Baseline 视觉节点发布

| Topic | Type | 说明 |
| --- | --- | --- |
| `/kele/detections` | `vision_msgs/msg/Detection3DArray` | 可乐检测结果，位姿为世界坐标系 `frame_id=world` |
| `/kele/result_image` | `sensor_msgs/msg/Image` | 检测可视化图，仅调试用 |

`/joint_states` 关节顺序如下：

```text
slide_joint
head_yaw_joint
head_pitch_joint
left_arm_joint1
left_arm_joint2
left_arm_joint3
left_arm_joint4
left_arm_joint5
left_arm_joint6
left_arm_eef_gripper_joint
right_arm_joint1
right_arm_joint2
right_arm_joint3
right_arm_joint4
right_arm_joint5
right_arm_joint6
right_arm_eef_gripper_joint
```

夹爪命令中 `gripper=1.0` 表示张开，baseline 抓可乐时使用 `gripper=0.08` 闭合。

## 参数说明

| 参数 | 推荐值 | 含义 |
| --- | --- | --- |
| `ROS_DOMAIN_ID` | `99` | ROS2 通信域，server/client 必须一致 |
| `RMW_IMPLEMENTATION` | `rmw_cyclonedds_cpp` | ROS2 RMW 实现 |
| `MUJOCO_GL` | `glfw` | 图形窗口运行；无头环境可改为 `egl` |
| `SUPERMARKET_HEADLESS` | `0` | `0` 显示窗口，`1` 无头运行 |
| `SUPERMARKET_ENABLE_RENDER` | `1` | 发布相机 RGB-D 数据 |
| `SUPERMARKET_USE_GS` | `1` | 启用 3D Gaussian Splatting 渲染 |
| `SUPERMARKET_RANDOMIZE` | `1` | 是否随机打乱 45 个商品位置 |
| `SUPERMARKET_SEED` | 可选 | 指定随机种子，使随机布局可复现；baseline 验证使用 `11` |
| `SUPERMARKET_ENABLE_SCORE` | `1` | 启用裁判计分和 `/referee/*` 话题 |

主要文件：

```text
examples/supermarket_sorting/supermarket_sorting_server.py   # 仿真 server
examples/supermarket_sorting/supermarket_sorting_client.py   # baseline 控制 client
examples/supermarket_sorting/perception/kele_detect.py       # baseline 可乐视觉检测
examples/supermarket_sorting/retail_competition_layout.json  # 货架布局和商品元数据
```
