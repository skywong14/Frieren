# Frieren LoRA Final Experiment Plan

本文件是后续实验的执行协议。后续训练、推理、结果分析都尽量按这里固定的实验组、prompt、seed 和评价标准进行，避免不同实验之间因为临时改设置而不可比。

## 1. 实验目标

本项目最终关注两个问题：

1. 小规模 Frieren 数据集经过 SDXL DreamBooth-style LoRA 微调后，是否能显著提升角色身份一致性？
2. 面对不同生成目标时，数据划分和 caption 组织方式会如何影响模型能力？

生成目标分三类：

- **2D Frieren**：标准动漫角色生成，重点看身份一致性和画面质量。
- **2D Frieren in Realistic World**：2D 动漫角色进入真实/摄影感背景，重点看风格混合能力。
- **3D / Semi-realistic Frieren**：半写实或 3D 化，重点看风格迁移时角色身份是否崩坏。

## 2. 最终实验组

最终不做大规模超参消融，而是围绕少量代表性实验展开。

| ID | 名称 | 数据 | Caption | 状态 | 作用 |
| --- | --- | --- | --- | --- | --- |
| B0 | Base SDXL | 无微调 | 无 | 不训练 | 无微调 baseline |
| L80 | 80-image structured LoRA | `frieren_hd_single80_v1` | 当前结构化/半结构化 caption | 已训练 | 当前候选最佳 LoRA |
| L100 | 100-image structured LoRA | `frieren_hd_all100_v1` | 沿用已有结构化 caption | 待训练 | 和 L80 比较，决定最终最佳 LoRA |
| L80-simple | 80-image simple-caption LoRA | `frieren_hd_single80_simple_caption_v1` | 简单固定 caption | 待训练 | 代表性 caption 对比 |

其中 L100 默认由 80 张单角色图加 20 张多人/复杂候选图组成。它不是“更多数据一定更好”的实验，而是用来观察额外复杂样本是否提升泛化，或是否污染角色身份。

## 3. 主要对比问题

### 3.1 微调有效性

对比：

- B0 vs L80
- B0 vs L100

回答：

- 原始 SDXL 是否能直接生成稳定 Frieren？
- LoRA 是否提升银发、精灵耳、服装、角色气质等身份特征？

### 3.2 数据划分探索

对比：

- L80 vs L100

回答：

- 干净单角色数据是否更利于身份一致性？
- 加入 20 张复杂/多人图后，是否提升现实场景或 3D/半写实场景中的泛化？
- L100 是否引入额外人物、构图混乱或身份漂移？

最终最佳 LoRA 在 L80 和 L100 中选择。选择依据不是单一画质，而是三类生成任务下的综合表现。

### 3.3 Caption 组织探索

对比：

- L80 vs L80-simple

回答：

- 结构化 caption 是否比简单 caption 更能保留角色细节？
- 简单 caption 是否更容易让触发词绑定完整身份？
- 哪种 caption 对动作、场景、风格迁移更稳定？

## 4. 固定评测 Prompt

统一 prompt bank 存放在：

`configs/prompt_banks/frieren_eval_v1.yaml`

prompt 分为三组：

| 组别 | Prompt ID | 目的 |
| --- | --- | --- |
| 2D Frieren | `p01_2d_portrait`, `p02_2d_full_body`, `p03_2d_spellcasting` | 标准角色身份、构图和动作 |
| 2D in realistic world | `p04_real_train_station`, `p05_real_cafe` | 现实背景中的 2D 角色融合 |
| 3D / semi-realistic | `p06_semireal_portrait`, `p07_3d_render` | 风格迁移和身份保持 |

每个模型对每个 prompt 生成 4 张图，固定 seed。默认采样参数：

| 参数 | 默认值 |
| --- | --- |
| Resolution | `768 x 768` |
| Inference steps | `30` |
| Guidance scale | `7.5` |
| Samples per prompt | `4` |
| Seed base | `42` |
| LoRA scale | `1.0` |

## 5. 评价标准

每组结果使用 1 到 5 分主观评分。建议每个 prompt 组单独评分，再汇总平均。

| 维度 | 说明 |
| --- | --- |
| Identity Consistency | 是否像 Frieren，是否保留银发、精灵耳、白金长袍、角色气质 |
| Prompt Following | 是否遵循动作、场景、构图和风格描述 |
| Image Quality | 清晰度、脸部、手部、身体结构、噪声、伪影 |
| Style Transfer | 是否能在现实背景、半写实或 3D 风格中保持目标风格 |
| Failure Rate | 4 张结果中明显崩坏、多人污染、身份漂移的比例 |

推荐报告中使用表格：

| Model | 2D Identity | Realistic-world Mixing | 3D/Semi-realistic | Image Quality | Failure Notes |
| --- | --- | --- | --- | --- | --- |
| B0 | | | | | |
| L80 | | | | | |
| L100 | | | | | |
| L80-simple | | | | | |

## 6. 推荐执行顺序

### Step 1: 准备派生数据集

```bash
python3 scripts/prepare_experiment_datasets.py --overwrite
```

生成：

- `data/datasets/frieren_hd_all100_v1`
- `data/datasets/frieren_hd_single80_simple_caption_v1`

### Step 2: 跑 L100 训练

```bash
scripts/train_sdxl_lora.sh configs/experiments/train_l100_structured.yaml
```

### Step 3: 跑 L80-simple 训练

```bash
scripts/train_sdxl_lora.sh configs/experiments/train_l80_simple_caption.yaml
```

### Step 4: 对所有模型跑统一评测

B0:

```bash
python3 scripts/eval_prompt_bank.py \
  --config configs/experiments/train_l80_structured_existing.yaml \
  --prompt-bank configs/prompt_banks/frieren_eval_v1.yaml \
  --model-label B0_base_sdxl
```

L80:

```bash
python3 scripts/eval_prompt_bank.py \
  --config configs/experiments/train_l80_structured_existing.yaml \
  --prompt-bank configs/prompt_banks/frieren_eval_v1.yaml \
  --lora-dir outputs/train/frieren_sdxl_lora_hd80_single_full_v1 \
  --model-label L80_structured
```

L100:

```bash
python3 scripts/eval_prompt_bank.py \
  --config configs/experiments/train_l100_structured.yaml \
  --prompt-bank configs/prompt_banks/frieren_eval_v1.yaml \
  --lora-dir outputs/train/frieren_sdxl_lora_hd100_full_structured_v1 \
  --model-label L100_structured
```

L80-simple:

```bash
python3 scripts/eval_prompt_bank.py \
  --config configs/experiments/train_l80_simple_caption.yaml \
  --prompt-bank configs/prompt_banks/frieren_eval_v1.yaml \
  --lora-dir outputs/train/frieren_sdxl_lora_hd80_single_simple_caption_v1 \
  --model-label L80_simple
```

### Step 5: 选择最佳 LoRA

只在 L80 和 L100 中选择最终最佳 LoRA。

建议判断逻辑：

- 如果 L100 在现实世界和 3D/半写实 prompt 上明显更强，且 2D 身份没有明显下降，选择 L100。
- 如果 L100 引入多人污染、服装漂移、身份不稳定，而 L80 更干净稳定，选择 L80。
- 如果二者接近，优先选择失败率更低的一组。

### Step 6: 可选 LoRA scale sweep

只对最终最佳 LoRA 做，避免工作量膨胀。

```bash
python3 scripts/eval_prompt_bank.py \
  --config configs/experiments/train_l80_structured_existing.yaml \
  --prompt-bank configs/prompt_banks/frieren_eval_v1.yaml \
  --lora-dir outputs/train/frieren_sdxl_lora_hd80_single_full_v1 \
  --model-label best_lora_scale_sweep \
  --lora-scales 0.6 0.8 1.0 1.2
```

## 7. 报告建议结论框架

最终报告中实验结论可以组织为：

1. **LoRA 微调有效**：相较 B0，L80/L100 明显提升 Frieren 身份一致性。
2. **数据更多不一定更好**：L100 可能提升复杂场景泛化，也可能引入多人污染和身份漂移。
3. **Caption 影响可控性**：结构化 caption 与简单 caption 在身份绑定和 prompt 可控性之间可能存在取舍。
4. **2D 到现实/3D 是主要难点**：2D 标准生成最稳定，现实背景混合和 3D/半写实最容易暴露身份漂移。
5. **最佳 LoRA 的选择应服务于任务目标**：如果报告重点是稳定角色生成，优先身份一致性；如果重点是风格迁移，适当考虑 L100 的泛化能力。

