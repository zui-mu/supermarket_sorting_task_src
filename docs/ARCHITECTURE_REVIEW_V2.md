# Supermarket Sorting V2 Architecture Review

Date: 2026-08-06

## 中文执行摘要

这次先冻结运行代码，不再继续调整抓取偏移量和卡死恢复时间。当前效果反复变差的根本原因不是某一个参数不对，而是系统的执行顺序和模块边界不适合 V2 正式赛：

1. 正式赛每局只有 5 个匿名订单，并不是运完 45 件商品。每成功完成一个“货架抓取 -> 完整送达”循环得 12 分，因此首先要追求一个循环可重复成功，再扩展到 5 个订单。
2. 当前程序收到订单后，会把每个目标类别和 45 个货位组合成候选任务，最多生成 225 个猜测，再先选货位、后去检测。这是机器人在货架间折返、频繁换目标和盯空位的直接原因。正确顺序应为“扫描并建立商品-货位地图 -> 从已确认商品中选目标 -> 锁定目标执行”。
3. 当前正式可用的感知只有单类可乐 YOLO/Blob 思路；测试时效果较好的 `gt` 和 `runtime_layout.json` 属于 Server 真值，正式规则禁止依赖。需要建设 9 类商品检测、ArUco 货位关联和跨循环库存记忆。
4. 当前抓取在头部相机定位一次后，使用固定腕部姿态和 IK，再让底盘向前推进，最后几厘米没有腕部相机闭环。不同物体只改平移偏移，无法从物理上覆盖圆柱、球体、楔形盒和宽 17.2 cm 的纸巾盒。
5. 当前导航是 A* 航点跟踪加手写激光阈值和倒车/旋转恢复，没有持续局部代价地图、完整机器人外形碰撞检查和统一的局部控制器，因此路径跟踪与避障转向会互相争夺控制权，表现为摇头、犹豫和反复转圈。
6. 当前抓取、放置和掉落判断高度依赖 `/referee/state`，但它不在官方公开传感器接口表中。正式程序必须用夹爪状态、RGB-D/腕部图像和目标是否随夹爪运动来独立验证，裁判状态只能做离线测试评分。
7. 建议引入的成熟基础设施是 Nav2、BehaviorTree.CPP、MoveIt 2/Servo 的思想和组件，而不是直接复制某个不同机器人的完整仓库。`warehouse_AMR` 适合参考节点拆分和数据流；Contact-GraspNet 等研究型抓取网络放到多类感知和标定稳定之后评估。
8. 新系统应与旧客户端并行建设：先做可重复测量，再单独验收导航，再验收合法感知，再逐类验收静态抓取，最后才跑一个完整订单和五订单任务。任何一层不过关，不进入下一层。

后续第一批工程任务应是：建立自动测试与指标报告（Gate 0），随后接入 Nav2 导航骨架并实现库存感知节点。旧代码保留为对照基线，不推翻 Git 历史。

## 1. Competition Objective

The official V2 task is a ten-minute mobile-manipulation mission with five
anonymous orders. Each order contains a product `kind`, but no shelf, slot,
ArUco ID, or world position. The robot must discover the inventory, pick one
item, deliver it immediately, and repeat.

The engineering objective is therefore not "attempt all 45 shelf objects". It
is to maximize complete and safe pick-deliver cycles:

- 12 points for each complete delivery, up to five;
- additional credit for efficient order selection and continuous operation;
- additional credit for smooth replanning and visual-servo grasping;
- penalties for shelf drops, transport drops, neighboring-item contact, and
  dangerous collisions.

One repeatable end-to-end delivery is more valuable than broad but unreliable
support for all products.

## 2. Current System

The current runtime is centered on a 2,400-line `PickPlaceClient`. It owns:

- base velocity control and acceleration limits;
- a grid A* planner and waypoint follower;
- lidar safety sectors, stuck detection, and recovery motion;
- target association and product-specific offsets;
- arm inverse kinematics and joint command interpolation;
- grasp, lift, carry, place, drop, and retry state transitions;
- referee-state interpretation.

Other current components are:

- `kele_detect.py`: head RGB-D deprojection with blob, single-class YOLO, or
  development-only ground-truth projection backends;
- `TaskManager`: expands official anonymous targets into slot candidates;
- `OrderScheduler`: ranks those candidates with hand-tuned bonuses;
- `SupermarketGridPlanner`: A* over static rectangles and current lidar points.

This was a reasonable way to extend the original fixed baseline quickly, but
it is not a stable architecture for the randomized V2 task.

## 3. Root Causes

### 3.1 Decision happens before perception

For an official anonymous target, the current `TaskManager` creates one task
for every public slot. Five orders can become 225 speculative tasks. The
scheduler chooses a slot before the inventory has been observed. A failed
detection then moves the robot to another guessed slot.

This directly produces long shelf traversals, target switching, repeated head
motion, and attempts at empty or stale positions.

The V2-required order is the opposite:

1. observe shelf stations;
2. associate product detections with ArUco slots;
3. maintain an inventory belief;
4. select an observed target using route cost and grasp risk;
5. lock that target until success or a clearly classified failure.

### 3.2 Perception is not competition-complete

The production detector and dataset scripts are primarily single-class
`kele`. The `gt` backend and `runtime_layout.json` are useful as test oracles,
but they read server truth and are not legal competition perception inputs.
Blob detection cannot provide reliable nine-class identity.

The current output is only a center point. It lacks the information needed for
shape-aware grasping: mask/extent, object orientation, depth quality,
visibility, neighboring clearance, and slot association confidence.

### 3.3 Grasping is mostly open loop

The head camera produces one target estimate, then the system commands a fixed
IK pose and drives the entire base forward for the final approach. There is no
wrist-camera visual servo during the last centimeters. Small base yaw, camera
calibration, depth-edge, and IK errors therefore become one-finger contact.

All products also share one wrist rotation. Translation offsets alone cannot
cover cylinders, wedges, spheres, and the 172 mm-wide tissue box. Product
handling should use a few geometric grasp families with distinct approach
orientations and pre-grasp constraints.

Grasp and placement success currently depend heavily on `/referee/state`.
That topic is not in the official public sensor interface table and must be a
development oracle, not a required runtime input.

### 3.4 Navigation lacks a complete local navigation stack

The custom A* planner treats a lidar scan as a short-lived set of points and
the follower combines heading control with reactive front-sector rules. It
does not maintain an obstacle costmap, model the full swept footprint of the
base plus carried arm, optimize local trajectories, or use a coherent
recovery behavior tree. The result is oscillation: the waypoint follower and
reactive obstacle turn can repeatedly command competing headings.

### 3.5 Failure handling is coupled to mission logic

Timeouts, retries, drop handling, navigation recovery, and task selection are
interleaved in the same tick function. A local recovery can therefore mutate
mission-level state. Repeated patches add state combinations that are hard to
test and make successful behavior depend on timing.

## 4. Target Architecture

Use separate ROS2 nodes with explicit contracts:

```text
/supermarket_sorting/task
          |
          v
  mission_orchestrator  <------ execution events / health
     |       |       |
     |       |       +----> place_server
     |       +------------> grasp_server
     +--------------------> navigation_server
          ^                       ^
          |                       |
  inventory_mapper         scan + odom + TF
          ^
          |
  multiclass_detector + ArUco + RGB-D
```

### 4.1 Perception and inventory

Build a nine-class detector or segmenter from legal RGB input. Combine its
mask/box with aligned depth to estimate object center, extent, and orientation.
Detect ArUco markers to associate each observation with a stable public slot.

Maintain an inventory record per slot:

```text
slot_id, kind distribution, pose, extent, confidence, timestamp,
graspability, occupied/removed/disturbed state
```

Never turn an unobserved slot into an executable pick task. Re-observe a
selected slot immediately before approach.

### 4.2 Mission and order scheduling

The orchestrator should use a behavior tree or a small hierarchical state
machine. Its stable mission states are:

```text
WAIT_TASK -> SCAN_INVENTORY -> SELECT_TARGET -> NAVIGATE_PICK
-> LOCALIZE_AND_GRASP -> VERIFY_GRASP -> NAVIGATE_DELIVERY
-> PLACE -> VERIFY_PLACE -> UPDATE_INVENTORY -> SELECT_TARGET
```

Target cost should be computed only for observed candidates:

```text
route_to_pick + route_pick_to_delivery
+ grasp_risk + visibility_risk + shelf_level_risk
```

Once local alignment begins, lock the target. Unlock only for an explicit
failure class such as target absent, unreachable, disturbed, or grasp failed.

### 4.3 Navigation

Use Nav2 concepts and, preferably, Nav2 itself:

- obstacle and inflation costmap layers from LaserScan;
- a footprint that accounts for the base and a conservative carried-arm
  envelope;
- global path planning and smoothing;
- a local trajectory controller with progress and goal checkers;
- collision monitor and behavior-tree recoveries;
- immediate replanning when the path is invalid, instead of repeated blind
  reverse/rotate cycles.

The official client image contains TF but not Nav2 packages. Build the Humble
Nav2 runtime into a derived team client image, or vendor only the required
packages. Do not rely on installing packages during a scored run.

Candidate configuration: Smac 2D or NavFn global planner, Regulated Pure
Pursuit first for predictable path tracking, costmap obstacle/inflation
layers, velocity smoother, collision monitor, and bounded backup/spin
recoveries. MPPI can be evaluated later if the base model and compute budget
justify it.

### 4.4 Manipulation

Retain the working MMK2 kinematics adapter initially, but isolate it behind a
`grasp_server` interface. Add collision and reachability checks before motion.

Use three or four grasp families rather than nine unrelated offset tables:

- upright cylinders: centered side pinch;
- wedge/box packages: mask/PCA center with shape-specific height;
- spheres: pinch slightly above the equator with a shallow lift;
- wide tissue box: rotated wrist for vertical/side pinch, subject to IK and
  shelf-clearance validation.

The last 5-10 cm must be closed loop. Use the right wrist camera to keep the
object center between the projected fingertip centers while commanding small
arm or base corrections. Approach slowly only in this terminal servo region;
normal navigation should remain fast.

Verify grasp without referee truth using at least two independent signals:

- gripper does not reach the fully closed empty value;
- target moves consistently with the wrist across several frames;
- target disappears from the original shelf slot after lift;
- optionally, actuator effort if it becomes available on a public topic.

Verify place by observing the product on the delivery platform and the gripper
returning to an empty state.

### 4.5 Development oracle separation

Keep `gt`, runtime layout, and referee feedback in a dedicated test-only
adapter. Production mode must fail startup if any illegal oracle is enabled.
The same mission code should run in both modes; only observers differ.

## 5. Open-Source Components

Adopt components by interface, not by copying a complete unrelated robot:

- Navigation2: production-grade ROS2 planning, control, costmaps, collision
  monitoring, recoveries, and behavior-tree navigation;
- BehaviorTree.CPP / Nav2 BT Navigator: mission and recovery orchestration;
- MoveIt 2 and MoveIt Servo: collision-aware arm planning and terminal Cartesian
  servo patterns, if an MMK2 URDF/SRDF and controller adapter are built;
- OpenCV ArUco: public-slot localization;
- Ultralytics YOLO segmentation/detection: nine-class product perception;
- Contact-GraspNet or GPD: later candidates for RGB-D grasp proposals after
  perception and calibration pass deterministic tests.

Useful architectural reference:
`tharittapol/warehouse_AMR` separates Nav2, RGB-D perception, TF transforms,
arm control, and orchestration. Its JetRover model and controllers are not
drop-in compatible with MMK2.

## 6. Iteration Plan and Gates

Freeze feature patches in the current monolithic client. Preserve it as a
baseline and build the replacement beside it.

### Gate 0: reproducible measurement

- fixed product and obstacle seed matrix;
- structured event log with timestamps and failure taxonomy;
- automatic metrics: path length, collisions, oscillation count, detection
  recall, grasp success, transport survival, placement success, cycle time;
- development oracle used only to score observations, never to command them.

Exit criterion: repeated runs produce comparable reports and video/log
artifacts.

### Gate 1: navigation only

- arm stowed and no products touched;
- start -> shelf station -> delivery -> shelf station;
- at least ten obstacle seeds;
- zero collision, zero unresolved stuck state, bounded path overhead, and no
  repeated left-right oscillation.

### Gate 2: legal inventory perception

- no `gt` backend or runtime layout;
- identify all nine kinds and associate detections to ArUco slots;
- evaluate across product seeds and all three levels;
- persist the map across delivery cycles.

Exit criterion: target-slot recall and precision are measured and sufficient
to avoid speculative shelf visits.

### Gate 3: stationary grasp families

- robot starts already aligned to one slot;
- test each grasp family and shelf level independently;
- no base navigation and no retry during the primary metric;
- report first-attempt success and neighboring-object disturbance.

Exit criterion: each required family reaches an agreed first-attempt success
rate before it enters full missions.

### Gate 4: one complete order

- legal perception, navigation, grasp verification, carry, place verification;
- no referee dependency;
- multiple product and obstacle seeds.

Exit criterion: one complete order is repeatable before enabling five orders.

### Gate 5: five-order mission

- inventory-based target selection;
- order optimization from delivery position;
- no inter-cycle pause above 15 seconds;
- ten-minute budget and official scoring report.

## 7. Immediate Decisions

1. Do not continue tuning `geometry_close_remaining`, product offsets, or blind
   recovery timings in the monolithic client until Gate 0 exists.
2. Keep the current working source and Git history; no code needs to be thrown
   away. Reuse the MMK2 FK/IK adapter, ROS topic bridge, public slot geometry,
   RGB-D deprojection, and grid planner as test references.
3. Build the new stack alongside the old one behind a separate launch script.
4. Implement measurement and legal perception before broad grasp tuning.
5. Treat Nav2 integration and the inventory mapper as the first structural
   changes because they remove the two largest observed failure classes:
   oscillating navigation and speculative target switching.

## 8. Evidence Sources

- Official V2 competition plan and image-change notice, local release dated
  2026-07-31 / 2026-08-01;
- official V2 README and ROS2 public topic table;
- current repository code and Git history;
- Navigation2 documentation: https://docs.nav2.org/
- Navigation2 source: https://github.com/ros-navigation/navigation2
- MoveIt 2 source: https://github.com/moveit/moveit2
- BehaviorTree.CPP: https://github.com/BehaviorTree/BehaviorTree.CPP
- Contact-GraspNet: https://github.com/NVlabs/contact_graspnet
- Warehouse AMR reference: https://github.com/tharittapol/warehouse_AMR
