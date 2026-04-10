# Frieren Character Diffusion Project

本仓库用于完成课程大作业：围绕动漫角色“芙莉莲（Frieren）”构建一个小规模角色个性化图像生成项目，重点研究：

1. 如何用约 100 张图像让扩散模型学会稳定生成“芙莉莲”这一角色；
2. 如何在保持角色身份的同时，实现不同画面风格，尤其是“现实风格背景 / 摄影感场景 + 2D 动漫芙莉莲”的混合效果。

本项目优先采用：
- **基础模型**：SDXL base 1.0
- **训练方法**：DreamBooth-style LoRA
- **训练框架**：Hugging Face Diffusers 官方脚本
- **可选增强**：IP-Adapter / ControlNet（推理阶段）

---

## 1. 项目目标

本项目不是单纯训练一个 LoRA，而是拆成两个子问题：

- **角色身份学习**：让模型学会“芙莉莲是谁”
- **跨风格生成控制**：让同一角色在动漫、半写实、摄影感场景中稳定出现

最终希望回答三个问题：

1. 只靠 prompt，底模能否稳定生成芙莉莲？
2. 结构化 caption 和更纯净的数据集，是否能提升角色一致性？
3. “现实风格 + 2D 芙莉莲”更适合在训练阶段学，还是在推理阶段通过 adapter / prompt 控制？

---

## 2. 推荐技术路线

### 主线
- 使用 **SDXL base 1.0** 作为底模
- 使用 **DreamBooth-style LoRA** 训练角色 LoRA
- 使用 **100 张芙莉莲图像** 构建训练集
- caption 采用“身份词 + 固定角色特征 + 动态特征”的结构化写法

### 增强线
- 在推理阶段引入 **IP-Adapter**，将真实摄影图的光影、场景氛围迁移到生成结果中
- 需要更强构图控制时，引入 **ControlNet**

### 为什么这样设计
- 训练阶段优先学习“角色身份”
- 推理阶段再做“现实化 / 风格控制”
- 这样更稳，也更利于写实验分析

---

## 3. 仓库建议结构

```text
frieren_project_starter/
├── README.md
├── STARTER_PROMPT.md
├── assets/
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── project_spec.md
│   └── implementation_plan.md
├── notebooks/
├── outputs/
└── scripts/
```

说明：
- `docs/`：项目文档与实验设计
- `data/raw/`：原始图像
- `data/processed/`：裁剪、清洗、caption 完成后的训练集
- `configs/`：训练配置
- `scripts/`：训练与推理脚本
- `outputs/`：checkpoint、生成图、对比网格图
- `notebooks/`：分析 loss、做结果可视化

---

## 4. 建议的最小可交付成果

### 第一阶段：跑通主线
- 整理 100 张训练图
- 完成 caption
- 用 Diffusers 官方脚本跑通一次 SDXL LoRA 训练
- 输出固定 prompt 下的生成结果

### 第二阶段：做分析
- 对比 baseline（未微调）与 LoRA
- 对比简单 caption 与结构化 caption
- 对比更纯净数据集与更丰富数据集

### 第三阶段：做亮点
- 最佳 LoRA + 现实摄影感 prompt
- 最佳 LoRA + IP-Adapter
- （可选）最佳 LoRA + ControlNet

---

## 5. 建议的开发顺序

1. 建立数据标准与文件命名规则
2. 整理候选图像并完成清洗
3. 编写 caption / metadata
4. 跑通训练脚本
5. 固定验证 prompt，定期采样
6. 选择最佳 checkpoint，做推理增强实验
7. 汇总生成图、loss 曲线与失败案例
8. 写报告

---

## 6. 当前建议优先做的文件

初始化 repo 后，优先补充：

- `configs/train_sdxl_lora.yaml`
- `scripts/train_sdxl_lora.sh`
- `scripts/infer_compare.py`
- `scripts/build_metadata.py`
- `data/processed/metadata.jsonl`

---

## 7. 一句话项目定位

> 一个围绕“芙莉莲”角色身份学习与跨风格可控生成的小型扩散模型课程项目，主线是 SDXL + DreamBooth-style LoRA，亮点是“现实风格场景 + 2D 动漫角色”的混合生成。
