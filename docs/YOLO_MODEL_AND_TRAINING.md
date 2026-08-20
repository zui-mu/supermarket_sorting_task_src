# YOLO 9 类识别模型：训练说明 + 获取方式（2026-08-20）

## 为什么需要这个模型
官方正式比赛任务只给商品 `kind`（如 `kele`、`zhijin`），不告诉位置。
机器人必须靠**相机视觉识别**货架上是哪种商品，才能建立"kind → 货位"库存表去抓取。
这个 9 类多分类 YOLO 模型就是识别的核心。

## 模型信息
- 文件：`examples/supermarket_sorting/perception/checkpoints/supermarket_multiclass.pt`
- 大小：约 21.5 MB（Git LFS / 单独分发）
- 类别（9 类，顺序固定）：
  ```
  sanmingzhi, heweidao, shupian, zhijin, maidong, kele, kouxiangtang, pingguo, chengzi
  ```
- 验证精度：`val mAP50 = 0.989, mAP50-95 = 0.918`

> **注意**：`.pt` 权重被 `.gitignore` 排除，**没有上传 GitHub**（22MB 不适合普通 git，且审计建议大模型单独分发）。团队同学需要**单独拷贝这个文件**（或按下方步骤自己训练）。

## 训练环境（本机实测可行）
用**官方 `supermarket_sorting:server` 容器**（不是 client！client 缺 EGL 渲染库）：
- 已有：Ubuntu22.04 + Python3.10 + CUDA12.8 + PyTorch2.7 + Ultralytics + OpenCV + EGL(server有)
- 命令（在仓库根目录 + 容器内 /workspace/baseline）：
```bash
# 1. 生成数据集（本地实测 500帧=1000图，~10 分钟）
cd /workspace/baseline/examples/supermarket_sorting
MUJOCO_GL=egl python3 perception/gen_dataset.py --frames 500 --variants 1 --pose-mode wide --label-mode geometry --overwrite

# 2. 训练（20 epochs about 25 分钟 on RTX4070）
cd /workspace/baseline/examples/supermarket_sorting
python3 perception/train_yolo.py --epochs 20 --batch 1 --imgsz 640 --device 0

# 3. 确认权重
ls -lh perception/checkpoints/supermarket_multiclass.pt
```
- 训练完自动复制 best 权重到 checkpoints/。

## 如何验证模型
```bash
# 启动正式模式感知（YOLO 后端）
SUPERMARKET_DETECT_BACKEND=yolo ./scripts/run_v2_perception.sh
# 查看识别输出
ros2 topic echo /supermarket_sorting/detections --once
```

## 更多数据（可选，提升精度）
- 当前 500 帧 × 2 = 1000 图 / 2053 框，mAP50-95 0.918
- 想更高：`--frames 1000 --variants 2` + `--epochs 60`（RTX4070 约 2 小时）
- 数据集生成后 ~100MB，可自行在本机生成（不需要共享）

## 团队同步方式
1. `git pull` 拿到最新代码（含本说明 + z 标定修复）
2. 单独拷贝权重文件 `supermarket_multiclass.pt`（本机路径：
   `E:\summer\jiebang\supermarket_sorting_task_src\examples\supermarket_sorting\perception\checkpoints\`）
3. 或按上述步骤自己训练（数据在本机，命令已验证）
