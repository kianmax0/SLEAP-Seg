# SLEAP-Seg 标注指南

本指南帮助实验室研究员用 **LabelMe** 为小鼠视频帧制作实例分割标注，用于训练 YOLOv8-seg 模型。

---

## 目录

1. [整体流程](#整体流程)
2. [安装 LabelMe](#安装-labelme)
3. [第一步：抽帧](#第一步抽帧)
4. [第二步：标注](#第二步标注)
5. [标注规范](#标注规范)
6. [常见错误](#常见错误)
7. [导出与训练](#导出与训练)

---

## 整体流程

```
视频文件 → 抽帧脚本 → LabelMe 标注 → 转换+训练脚本 → 自定义 YOLO 模型
```

完成一次完整流程大约需要 **2-3 小时**（含 ~200 张图的标注时间）。

---

## 安装 LabelMe

建议在项目的 conda 环境中安装：

```bash
conda activate sleapSeg
pip install "labelme[ai]"
```

验证安装：

```bash
labelme --version
```

---

## 第一步：抽帧

使用项目提供的脚本从视频中自动抽帧：

```bash
conda activate sleapSeg

# 从单个视频抽帧，目标 200 张，保存到项目 data 目录
python scripts/extract_frames.py \
    --video /path/to/your_video.Avi \
    --target 200 \
    --output data/frames_to_label/

# 从整个文件夹抽帧（推荐，覆盖更多行为）
python scripts/extract_frames.py \
    --folder /path/to/video_folder/ \
    --target 200 \
    --prefer-clahe \
    --output data/frames_to_label/
```

完成后 `data/frames_to_label/` 文件夹中会有约 200 张 `.jpg` 图片。

> **帧覆盖要求**：确保至少 **30%** 的帧中两鼠存在接触或遮挡，这对训练鲁棒性最重要。

---

## 第二步：标注

### 2.1 启动 LabelMe

```bash
conda activate sleapSeg
labelme data/frames_to_label/
```

LabelMe 将打开，左侧是图片列表，右侧是标注区域。

### 2.2 使用 SAM 魔棒（推荐，最快）

1. 在菜单中选择 **Edit → Create AI-Polygon**
2. 左键点击小鼠身体 → SAM 自动生成轮廓
3. 鼠标右键或拖动控制点来微调多边形
4. 在弹出的标签框中输入 `mouse`，按 Enter 确认

每只鼠点击一次，约 3-5 秒即可完成一张图。

### 2.3 使用手动多边形（备用）

1. 菜单：**Edit → Create Polygons**（或按 `P`）
2. 沿小鼠身体边缘点击放置控制点（约 10-20 个点）
3. 双击或按 `Enter` 闭合多边形
4. 输入标签名 `mouse`

### 2.4 保存

每张图标注完成后按 `Ctrl+S` 保存（或勾选自动保存）。LabelMe 会在同目录生成同名 `.json` 文件。

---

## 标注规范

### 标签名称

**所有小鼠统一使用单一标签：`mouse`**

不需要区分个体（哪只是 1 号、哪只是 2 号）。YOLO 只负责"找到鼠在哪"，个体 ID 由下游的 ByteTrack + Re-ID 追踪层自动分配和维持。

### 标注范围

- 包含整个身体（从鼻子到尾根）
- **不**包含尾巴（尾巴细长，会导致分割不稳定）
- 轮廓贴合毛发边缘，误差在 5px 以内

### 遮挡帧标注

| 情况 | 处理方式 |
|------|----------|
| 两鼠分开，各自清晰 | 分别用魔棒标注两个 `mouse` |
| 两鼠轻微接触 | 仍分别标注两个 `mouse`，轮廓可以相切 |
| 两鼠完全重叠、无法区分 | 用魔棒框出**整个重叠区域**，标注为**一个** `mouse` |
| 分离后 | 重新标注两个单独的 `mouse` |

> **说明**：两鼠完全重叠时标注为一个整体是正确做法。YOLO 学到的是"有鼠存在于此区域"；遮挡解除后重新出现两个独立轮廓时，Re-ID 模块会利用外观特征自动恢复各自的 ID。

### 推荐标注数量

| 场景 | 帧数 |
|------|------|
| 正常行走/探索（两鼠分离） | 80 帧 |
| 轻微接触 | 50 帧 |
| 明显遮挡/完全重叠 | 50 帧 |
| 静止 | 20 帧 |
| **总计** | **≥200 帧** |

---

## 常见错误

### ❌ 错误 1：使用了 `mouse_1` / `mouse_2` 等带编号的标签

```
mouse_1, mouse_2, Mouse, MOUSE  ← 都不对
```
**解决**：只使用小写的 `mouse`，所有鼠统一标签。

---

### ❌ 错误 2：两鼠完全重叠时强行标注两个多边形

完全重叠时，SAM 魔棒无法分辨两只鼠的边界，强行标注反而会产生错误的训练数据。此时标注一个整体轮廓即可。

---

### ❌ 错误 3：标注过于粗糙

轮廓离身体边缘太远会让模型学到背景噪声。目标是贴合毛发边缘，误差在 5px 以内。

---

### ❌ 错误 4：标注了尾巴

尾巴非常细，容易导致分割不稳定。轮廓应在尾根处（臀部）结束。

---

## 导出与训练

标注完成后，`data/frames_to_label/` 目录中会有：
- 图片文件：`*.jpg`
- 标注文件：`*.json`（与图片同名）

运行一键训练脚本：

```bash
conda activate sleapSeg

python scripts/train_yolo.py \
    --labels data/frames_to_label/ \
    --output runs/mice_seg/ \
    --base-model models/yolov8n-seg.pt \
    --epochs 100 \
    --device mps
```

训练完成后，脚本会自动：
1. 将最佳模型复制到 `models/yolov8_mice.pt`
2. 更新 `config/default.yaml` 中的模型路径

验证效果：

```bash
python scripts/visualize.py \
    --video 你的视频.mp4 \
    --skip-sleap
```

---

## 多实验室注意事项

如果你是**不同实验室**或使用**不同笼子/光照**条件：

1. 标注时，图片应来自你自己实验室的视频
2. 训练时，使用 `config/lab_whitebox.yaml` 或 `config/lab_blackbox.yaml` 作为基础配置
3. 建议每次新实验条件至少标注 50 张图做增量微调

---

## 需要帮助？

- [LabelMe 官方文档](https://github.com/labelmeai/labelme)
- [YOLOv8 文档](https://docs.ultralytics.com/tasks/segment/)
- 项目 GitHub: https://github.com/kianmax0/SLEAP-Seg
