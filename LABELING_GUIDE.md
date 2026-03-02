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

完成一次完整流程大约需要 **3-4 小时**（含 ~200 张图的标注时间）。

---

## 安装 LabelMe

建议在项目的 conda 环境中安装：

```bash
conda activate sleapSeg
pip install labelme
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

# 从单个视频抽帧，目标 200 张
python scripts/extract_frames.py \
    --video /Users/kuhn/Desktop/sleap/nature_social_data/fig01_behavior/00_RAW/FVB/20230807171613_fvb_01_day0.Avi \
    --target 200 \
    --output frames_to_label/

# 从整个文件夹抽帧（推荐，覆盖更多行为）
python scripts/extract_frames.py \
    --folder /Users/kuhn/Desktop/sleap/nature_social_data/fig01_behavior/00_RAW/FVB/ \
    --target 200 \
    --prefer-clahe \
    --output frames_to_label/
```

完成后 `frames_to_label/` 文件夹中会有约 200 张 `.jpg` 图片。

> **帧覆盖要求**：确保至少 **30%** 的帧中两鼠存在接触或遮挡（这对训练鲁棒性最重要）。

---

## 第二步：标注

### 2.1 启动 LabelMe

```bash
conda activate sleapSeg
labelme frames_to_label/
```

LabelMe 将打开，左侧是图片列表，右侧是标注区域。

### 2.2 使用 SAM 魔棒（推荐，最快）

1. 确保已安装 SAM 支持（LabelMe 1.0+）：
   ```bash
   pip install "labelme[ai]"
   ```
2. 在菜单中选择 **Edit → Create AI-Polygon**
3. 左键点击鼠标身体 → 自动生成轮廓
4. 鼠标右键或调整控制点来修正多边形
5. 在弹出的标签框中输入标签名（见下方规范）

### 2.3 使用手动多边形（备用）

1. 菜单：**Edit → Create Polygons**（或按 `P`）
2. 点击鼠标轮廓，沿身体边缘放置控制点（约 10-20 个点）
3. 双击或按 `Enter` 闭合多边形
4. 输入标签名

### 2.4 保存

每张图标注完成后按 `Ctrl+S` 保存（或勾选自动保存）。LabelMe 会在同目录生成 `.json` 文件。

---

## 标注规范

### 标签名称

| 个体 | 标签 | 说明 |
|------|------|------|
| 第一只鼠 | `mouse_1` | 通常是体型较大或行为发起方 |
| 第二只鼠 | `mouse_2` | 另一只鼠 |

> **重要**：同一视频中要保持个体标签一致。如果第一帧中体型大的是 `mouse_1`，后续帧也应如此。

### 标注范围

- 包含整个身体（从鼻子到尾根）
- **不**包含尾巴（尾巴细长，会增加模型误检）
- 轮廓贴合毛发边缘，不要太松

### 遮挡帧标注（关键！）

遮挡帧是标注中最重要的部分：

| 情况 | 处理方式 |
|------|----------|
| 一只鼠部分被遮挡 | 标注可见部分的轮廓（不要"猜测"被遮住的部分） |
| 两鼠完全叠压 | 分别标注各自能看到的区域，允许重叠 |
| 完全看不见 | 跳过该个体（不标注） |

### 推荐标注数量

| 场景 | 帧数 |
|------|------|
| 正常行走/探索 | 70 帧 |
| 轻微接触 | 50 帧 |
| 明显遮挡 | 60 帧 |
| 静止 | 20 帧 |
| **总计** | **≥200 帧** |

---

## 常见错误

### ❌ 错误 1：标签名称不统一

```
mouse1, Mouse_1, mouse_1, MOUSE_1  ← 都被视为不同类别
```
**解决**：只使用 `mouse_1` 和 `mouse_2`（全小写，下划线）。

---

### ❌ 错误 2：遮挡帧中只标注了一只鼠

如果两鼠相互遮挡，必须尝试标注两只。哪怕其中一只只能看到一小部分也要标注。

---

### ❌ 错误 3：标注过于粗糙

轮廓离身体边缘太远会让模型学到背景噪声。目标是轮廓贴合实际毛发边缘，误差在 5px 以内。

---

### ❌ 错误 4：标注了尾巴

尾巴非常细，容易导致分割不稳定。轮廓应在尾根处（臀部）结束。

---

## 导出与训练

标注完成后，在 `frames_to_label/` 目录中会有：
- 图片文件：`*.jpg`
- 标注文件：`*.json`（与图片同名）

运行一键训练脚本：

```bash
conda activate sleapSeg

python scripts/train_yolo.py \
    --labels frames_to_label/ \
    --output runs/mice_seg/ \
    --base-model models/yolov8n-seg.pt \
    --classes mouse_1 mouse_2 \
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
- 项目 GitHub: https://github.com/kuhn/SLEAP-Seg
