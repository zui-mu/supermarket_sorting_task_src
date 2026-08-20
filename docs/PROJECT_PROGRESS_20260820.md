# 项目当前进度 + 团队商量要点（2026-08-20）

## 一、已完成并验证（可用的能力）

### 1. 多轮基础改进（v70 基线：官方无头模式 4/5 满分 100）
- 抓取：闭爪时机修复、关节反馈证据（空合重试）、endpoint 0.042
- 配送：起步 stuck 误判修复、escape 僵局逃逸（长倒车+角度闭环 180° 掉头）、面包屑回退、持久障碍记忆
- 放置：5 个互异位置、POST_PLACE_EGRESS（后退0.25m+收臂后转身）
- 58 项单元测试全绿

### 2. YOLO 9 类识别模型（本轮重大新增，已落地）
- 环境：server 容器（有 EGL）
- 训练：500帧×2=1000图，val mAP50=0.989 / mAP50-95=0.918
- 权重：`supermarket_multiclass.pt`（21.5MB）
- 验证：正式匿名任务下能识别 zhijin（z 已修正到 0.879/接近 L2 中心）

### 3. 感知 z 标定修复（正式模式关键）
- 发现：YOLO RGB-D 检测 z 比真实中心低 ~0.4m（RGB 表面点 vs 物体中心）
- 修复：`surface_to_center_z` 各商品半高补偿（kele 0.0725、zhijin 0.05 等）
- 离线验证通过，已提交 commit `374cca6`

## 二、当前卡点（需团队商量）

### 正式匿名任务"货位关联"工作流待重构
- 现状：`task_manager` 用静态布局建 45 个货位搜索任务，但 server 随机化后商品搬走
- 问题：机器人扫静态预期的货位（如 A_L2_C1 预期 zhijin）却看不到，实际 zhijin 在别的货位
  → 检测的 zhijin 世界坐标与 search_slot 关联不上 → `assoc_reject` → 无法锁定抓取
- 应改：正式模式**靠 ArUco 定位 + 视觉识别 kind** 确认"这个货位上是什么"，不依赖静态布局坐标

### 本地 server 随机布局的 aruco/货位关联有 bug
- `runtime_layout.json` 里多个商品 aruco/shelf 标签错乱（正式 server 可能不同，需核实）

## 三、团队讨论问题（请筒靴一起定）

1. **优先做哪块？**
   - A. 重构正式模式货位关联（靠 ArUco 扫描识别，不依赖静态布局）← 建议优先
   - B. 多 seed 压测攒基线（GT 模式验证主流程稳定性）
   - C. 提升抓取/放置稳定性（已有不少修复，再精调）

2. **权重文件怎么共享？**
   - 21.5MB 的 `supermarket_multiclass.pt` 被 gitignore 排除、未上传 GitHub
   - 建议：单独拷贝 / 或用 Git LFS / 或团队各自训练
   - 已写说明：`docs/YOLO_MODEL_AND_TRAINING.md`

3. **正式模式 vs 开发模式怎么平衡？**
   - 开发/可视化用 GT 方便（但审计提醒别依赖）
   - 正式必须 YOLO + 不读 runtime_layout —— 已确认脚本保护正确

## 四、日志与文档
- 迭代记录：`docs/CODE_AUDIT_FIXES_20260812.md`
- YOLO 说明：`docs/YOLO_MODEL_AND_TRAINING.md`
- 最新提交记录见 git log
