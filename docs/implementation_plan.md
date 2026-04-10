# 项目文档二：实现路径与执行计划

## 1. 总体实现方案

本项目分为四个阶段：

1. 数据集构建
2. LoRA 训练与验证
3. 推理增强与风格混合实验
4. 结果整理与报告撰写

---

## 2. 阶段一：数据集构建

## 2.1 数据收集
目标收集 130~150 张候选图，最终筛选为 100 张训练图。

来源建议：
- 官方视觉图
- 动画截图
- 漫画彩页
- 高质量同人图

### 命名建议
统一命名为：

```text
frieren_0001.png
frieren_0002.png
...
```

---

## 2.2 数据清洗

清洗流程建议：
1. 人工去重
2. 删除极低分辨率图像
3. 删除水印严重图像
4. 删除角色设定不稳定图像
5. 删除多角色难裁剪图像

### 产物
- `data/raw/`：原始图像
- `data/processed/images/`：裁剪和清洗后的图像

---

## 2.3 caption / metadata 构建

### 方案 A：metadata.jsonl
适合 Diffusers。

每行示例：

```json
{"file_name": "frieren_0001.png", "text": "sks_frieren, elf girl, long silver hair, pointy ears, white and gold robe, holding a staff, calm expression, upper body, fantasy forest background"}
```

### 方案 B：每图一个 txt
适合 sd-scripts。

例如：
- `frieren_0001.png`
- `frieren_0001.txt`

---

## 3. 阶段二：LoRA 训练

## 3.1 推荐框架

### 主推荐
Hugging Face Diffusers 官方脚本：
- `train_dreambooth_lora_sdxl.py`

### 备选
kohya-ss / sd-scripts：
- `sdxl_train_network.py`

原则：
- 正式实验优先使用 Diffusers
- 标注、辅助尝试、快速对比可使用 sd-scripts

---

## 3.2 最小训练目标

先完成一条最小可行路线：

- 输入：100 张清洗后图像 + 结构化 caption
- 模型：SDXL base 1.0
- 方法：LoRA
- 输出：可通过触发词 `sks_frieren` 生成角色的 LoRA 权重

---

## 3.3 验证策略

建立固定验证 prompt 列表，每次训练后都跑相同 prompt。

建议分三类：

### A. 标准角色图
- 正脸、半身、全身
- 标准服装
- 标准法杖

### B. 动作 / 场景变化
- 行走、施法、远景
- 森林、黄昏、室内、雪景

### C. 跨风格测试
- 摄影感光影
- 现实风格背景
- 2D 角色 + 真实镜头语言

---

## 4. 阶段三：实验设计

## 4.1 Baseline
**E0：原始底模 + prompt**

目的：
验证底模是否已经足以稳定生成芙莉莲。

---

## 4.2 caption 消融
**E1：简单 caption**

示例：
```text
sks_frieren, elf girl, silver hair, pointy ears, white robe
```

**E2：结构化 caption**

示例：
```text
sks_frieren, elf girl, long silver hair, pointy ears, white and gold robe, holding a staff, calm expression, upper body, fantasy forest background
```

目的：
比较 caption 细粒度是否提升角色一致性与可控性。

---

## 4.3 数据集消融
**E3：Core-70 vs Mixed-100**

目的：
比较纯净数据集与更丰富数据集对 LoRA 训练效果的影响。

---

## 4.4 推理增强
**E4：最佳 LoRA + 风格控制**

包括：
- 最佳 LoRA + 写实 prompt
- 最佳 LoRA + IP-Adapter
- （可选）最佳 LoRA + ControlNet

目的：
验证“现实风格 + 2D 芙莉莲”是否更适合在推理阶段实现。

---

## 5. 阶段四：结果整理

## 5.1 需要保存的材料

- 训练配置文件
- loss 曲线
- 不同 checkpoint 采样图
- 固定 prompt 多组对比图
- 最佳实验结果图
- 失败案例图

---

## 5.2 建议输出形式

### 训练结果表
记录：
- 实验编号
- 数据集版本
- caption 方案
- 是否使用 adapter
- 主观评分
- 备注

### 可视化图组
- baseline vs LoRA
- simple caption vs structured caption
- Core-70 vs Mixed-100
- 普通 prompt vs IP-Adapter

---

## 6. 报告建议提纲

1. 引言
2. 任务定义
3. 数据集构建
4. 方法与实现
5. 实验设计
6. 结果分析
7. 局限性
8. 结论

---

## 7. 里程碑建议

### Milestone 1
完成数据收集、清洗、caption。

### Milestone 2
跑通第一版 LoRA。

### Milestone 3
完成 baseline/caption/dataset 三类对比。

### Milestone 4
完成现实风格混合实验。

### Milestone 5
完成报告与图表整理。

---

## 8. 下一步立即要做的事

1. 建 repo 并提交这 3 个文档
2. 补 `configs/` 和 `scripts/` 初版
3. 确定训练数据命名与 caption 规范
4. 整理首批 30 张高质量图，先跑通小样本实验
5. 验证脚本与采样流程稳定后，再扩展到完整 100 张
