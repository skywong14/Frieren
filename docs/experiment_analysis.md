# Frieren LoRA Experiment Analysis

本文件整理当前已经完成的统一评测结果，并给出后续报告写作可以直接采用的初版分析。当前分析基于统一 prompt bank 生成结果：

- Prompt bank: `configs/prompt_banks/frieren_eval_v1.yaml`
- Evaluation output: `outputs/experiments/frieren_eval_v1/`
- Contact sheets: `outputs/experiments/frieren_eval_v1/contact_sheets/`

## 1. 已完成实验

### 1.1 模型组

| ID | 模型 | 数据 | Caption | 训练状态 | 用途 |
| --- | --- | --- | --- | --- | --- |
| B0 | Base SDXL | 无 | 无 | 不训练 | 无微调 baseline |
| L80 | `frieren_sdxl_lora_hd80_single_full_v1` | 80 张单角色图 | 结构化/半结构化 caption | 已完成 | 现有候选最佳 LoRA |
| L100 | `frieren_sdxl_lora_hd100_full_structured_v1` | 80 张单角色图 + 20 张复杂/多人候选图 | 继承结构化 caption | 已完成 | 最终候选最佳 LoRA |
| L80-simple | `frieren_sdxl_lora_hd80_single_simple_caption_v1` | 80 张单角色图 | 统一简单 caption | 已完成 | Caption 组织方式对比 |

### 1.2 统一评测设置

统一评测脚本对每组模型使用完全相同的 prompt、seed 和采样参数。每个模型生成 7 个 prompt，每个 prompt 4 张图，总计 28 张图。

| 参数 | 值 |
| --- | --- |
| Resolution | `768 x 768` |
| Inference steps | `30` |
| Guidance scale | `7.5` |
| LoRA scale | `1.0` |
| Samples per prompt | `4` |
| Seed base | `42` |

Prompt 分为三类：

| 类别 | Prompt ID | 目的 |
| --- | --- | --- |
| 2D Frieren | `p01_2d_portrait`, `p02_2d_full_body`, `p03_2d_spellcasting` | 测试标准动漫角色身份、构图和动作 |
| 2D in realistic world | `p04_real_train_station`, `p05_real_cafe` | 测试 2D 角色进入真实/摄影感背景 |
| 3D / semi-realistic | `p06_semireal_portrait`, `p07_3d_render` | 测试半写实/3D 风格迁移时身份是否保持 |

## 2. 训练过程概览

三组 LoRA 使用相同训练超参：SDXL base、768 分辨率、LoRA rank 16、1200 steps、learning rate `1e-4`。

| 模型 | Initial loss | Final loss | Min loss | 观察 |
| --- | ---: | ---: | ---: | --- |
| L80 | 0.153391 | 0.011553 | 0.001407 | 正常收敛 |
| L100 | 0.141887 | 0.011291 | 0.001372 | 正常收敛，final loss 略低 |
| L80-simple | 0.154425 | 0.011773 | 0.001431 | 正常收敛，final loss 略高 |

Loss 只能说明训练过程稳定，不能直接等价于图像质量。最终模型选择仍以统一评测图像中的角色一致性、prompt 遵循度、风格迁移能力和失败率为主。

## 3. 关键可视化结果

### 3.1 标准 2D 角色头像

![p01 2D portrait](../outputs/experiments/frieren_eval_v1/contact_sheets/p01_2d_portrait_comparison.jpg)

观察：

- B0 可以生成白发精灵女性，但更像通用 fantasy elf，不稳定地复现 Frieren 的具体动画角色特征。
- L80、L100 和 L80-simple 都显著强化了 Frieren 身份，包括银白长发、精灵耳、白金长袍、红色耳坠和接近动画截图的脸部风格。
- L80 与 L100 在该 prompt 上非常接近，说明 80 张单角色图已经足以学习稳定 2D Frieren 身份。
- L80-simple 也能学到身份，但部分样本的构图和脸部稳定性略弱。

初步结论：

> LoRA 微调对角色身份学习非常有效；在标准 2D 头像任务中，L80 与 L100 均明显优于原始 SDXL。

### 3.2 2D 全身 fantasy 场景

![p02 2D full body](../outputs/experiments/frieren_eval_v1/contact_sheets/p02_2d_full_body_comparison.jpg)

观察：

- B0 的画面更写实、更 fantasy illustration，但角色身份仍然偏泛化。
- L80 与 L100 都能生成更接近 Frieren 的动画角色，并稳定保留白金长袍。
- L80-simple 更容易出现背面或侧后方视角，说明简单 caption 对构图和视角的约束较弱。
- L100 在部分样本中人物比例和背景融合略自然，但优势不大。

初步结论：

> 结构化 caption 对全身构图和角色可控性有帮助；简单 caption 虽能绑定身份，但对 `full body`、正面视角和场景细节的控制较弱。

### 3.3 2D 施法动作

![p03 2D spellcasting](../outputs/experiments/frieren_eval_v1/contact_sheets/p03_2d_spellcasting_comparison.jpg)

观察：

- B0 能生成施法氛围，但角色身份仍偏通用白发精灵。
- LoRA 模型能够稳定保持 Frieren 服装和动画风格。
- L80、L100 都能响应施法动作，但动作多表现为持杖、火光或魔法光效；复杂手部和法杖结构仍有一定不稳定。
- L80-simple 在部分 seed 中出现背面或侧面，动作与角色身份虽然存在，但 prompt 遵循度略弱。

初步结论：

> LoRA 提升了身份一致性，但复杂动作仍是难点。结构化 caption 相比简单 caption 对动作可控性更有利。

### 3.4 现实车站中的 2D Frieren

![p04 realistic train station](../outputs/experiments/frieren_eval_v1/contact_sheets/p04_real_train_station_comparison.jpg)

观察：

- B0 能较好生成真实车站背景，但角色偏写实白发精灵，不像明确的 2D Frieren。
- L80 和 L100 都能生成“2D 动漫角色 + 现实车站背景”的混合效果。
- L100 在背景透视、站台元素和人物融合上略自然；没有明显因为加入 20 张复杂样本而产生多人污染。
- L80-simple 也能完成任务，但部分样本人物方向和构图控制偏弱。

初步结论：

> 现实背景混合任务中，LoRA 能稳定保持 2D 角色身份；L100 的复杂样本可能对现实/复杂场景泛化有轻微帮助。

### 3.5 现实咖啡馆中的 2D Frieren

![p05 realistic cafe](../outputs/experiments/frieren_eval_v1/contact_sheets/p05_real_cafe_comparison.jpg)

观察：

- B0 的咖啡馆氛围自然，但角色仍不是稳定 Frieren。
- L80 和 L100 都较好实现了 2D 动漫角色坐在现实/摄影感咖啡馆中的效果。
- L100 的餐桌、杯子、咖啡馆光照更稳定一些；L80 的角色身份也非常强。
- L80-simple 有较强身份绑定，但部分表情、视角和物体交互略弱。

初步结论：

> 2D-in-realistic-world 是本项目最有展示价值的方向之一。L80 和 L100 都能完成，L100 在复杂背景中的综合表现略占优。

### 3.6 半写实 Frieren

![p06 semi-realistic portrait](../outputs/experiments/frieren_eval_v1/contact_sheets/p06_semireal_portrait_comparison.jpg)

观察：

- B0 能生成半写实精灵肖像，但身份更像普通 fantasy elf。
- L80、L100、L80-simple 仍明显倾向动画截图风格，说明 LoRA 强烈绑定了 2D 角色风格。
- L100 在某些样本中相较 L80 略微更接近半写实光影，但整体仍不是严格的 realistic portrait。

初步结论：

> 当前 LoRA 更擅长保持 2D Frieren 身份，而不是彻底转换为半写实风格。身份学习和风格迁移之间存在张力。

### 3.7 3D 风格 Frieren

![p07 3D render](../outputs/experiments/frieren_eval_v1/contact_sheets/p07_3d_render_comparison.jpg)

观察：

- B0 更容易响应 3D/写实 fantasy render，但角色身份弱。
- LoRA 模型能保持 Frieren 身份，但 3D 化程度有限，很多结果仍像 2D 动画截图。
- L100 和 L80 在 3D prompt 上差距不大；L100 的部分光影和体积感略强。
- L80-simple 在该任务中出现更明显的风格和构图不稳定。

初步结论：

> 3D/半写实生成是当前方案的主要短板。LoRA 对角色身份绑定很强，但也把训练集中 2D 动画风格一起绑定进模型，导致 prompt 很难完全把角色推向 3D。

## 4. 横向对比结论

### 4.1 B0 vs LoRA

| 对比 | 结论 |
| --- | --- |
| 身份一致性 | LoRA 明显优于 B0。B0 多生成通用白发精灵，而非稳定 Frieren。 |
| Prompt 遵循 | B0 对通用场景和风格词响应较好，但角色身份弱；LoRA 在保持身份的同时仍能响应主要场景。 |
| 风格迁移 | B0 更容易变成写实/3D，但无法保持 Frieren 身份；LoRA 保持身份强，但风格迁移受限。 |

可以写入报告的结论：

> 原始 SDXL 具备生成白发精灵和 fantasy 场景的能力，但缺乏特定角色身份知识。LoRA 微调显著提升了 Frieren 的身份一致性，说明小规模角色数据集足以完成角色个性化生成。

### 4.2 L80 vs L100

| 对比 | 观察 |
| --- | --- |
| 2D 角色 | 两者都很强，差距较小。 |
| 现实背景 | L100 略有优势，背景和人物融合更自然。 |
| 3D/半写实 | L100 部分样本光影和体积感略好，但整体仍受 2D 风格限制。 |
| 失败率 | 当前观察中 L100 没有明显多人污染。 |

可以写入报告的结论：

> 80 张干净单角色图已经足以学习稳定 Frieren 身份。加入额外 20 张复杂/多人候选图后，L100 没有明显损害角色一致性，并在现实背景和复杂场景泛化上表现出轻微优势。因此本文选择 L100 作为最终最佳 LoRA。

### 4.3 L80 structured vs L80-simple

| 对比 | 观察 |
| --- | --- |
| 身份绑定 | 两者都能学到 Frieren 身份。 |
| 构图控制 | 结构化 caption 更稳，L80-simple 更容易出现背面/侧面或视角偏差。 |
| 动作控制 | 结构化 caption 对施法、全身、场景等动态信息更友好。 |
| 风格迁移 | 简单 caption 没有明显提升 3D/半写实迁移，反而在可控性上略弱。 |

可以写入报告的结论：

> 简单 caption 可以让触发词绑定角色身份，但结构化 caption 对构图、动作和场景控制更有帮助。对于本项目这类需要比较 2D、现实背景和 3D prompt 的任务，结构化 caption 更适合作为主方案。

## 5. 当前最佳模型选择

当前建议选择：

> **L100 structured LoRA**：`outputs/train/frieren_sdxl_lora_hd100_full_structured_v1`

选择理由：

- 相较 B0，L100 明显提升 Frieren 身份一致性。
- 相较 L80，L100 在标准 2D 任务中没有明显退化。
- 相较 L80，L100 在现实车站、咖啡馆和部分半写实/3D prompt 上略有更好的复杂场景泛化。
- 相较 L80-simple，L100 的构图、动作和 prompt 遵循度更稳定。

需要谨慎表述的地方：

- L100 对 L80 的优势不是压倒性的，而是综合表现略优。
- 3D/半写实任务仍未完全成功，模型经常保持 2D 动画截图风格。
- 当前评价主要基于主观视觉比较，还没有使用 CLIP/DINO 等自动指标。

## 6. 建议补充实验

当前实验已经足够支撑课程报告。若时间允许，建议只补两个轻量实验，不建议再新增训练任务。

### 6.1 最佳 LoRA 的 scale sweep

目的：

- 观察 LoRA 强度对身份一致性和风格迁移的影响。
- 回答一个很实用的问题：`lora_scale=1.0` 是否是最佳推理设置？

推荐只对 L100 做：

```bash
conda activate frieren

python3 scripts/eval_prompt_bank.py \
  --config configs/experiments/train_l100_structured.yaml \
  --prompt-bank configs/prompt_banks/frieren_eval_v1.yaml \
  --lora-dir outputs/train/frieren_sdxl_lora_hd100_full_structured_v1 \
  --model-label L100_scale_sweep \
  --output-root outputs/experiments/frieren_eval_l100_scale_sweep \
  --lora-scales 0.6 0.8 1.0 1.2
```

如果想省时间，可以只生成每个 prompt 2 张：

```bash
conda activate frieren

python3 scripts/eval_prompt_bank.py \
  --config configs/experiments/train_l100_structured.yaml \
  --prompt-bank configs/prompt_banks/frieren_eval_v1.yaml \
  --lora-dir outputs/train/frieren_sdxl_lora_hd100_full_structured_v1 \
  --model-label L100_scale_sweep \
  --output-root outputs/experiments/frieren_eval_l100_scale_sweep \
  --num-images-per-prompt 2 \
  --lora-scales 0.6 0.8 1.0 1.2
```

预期观察：

- `0.6`：风格迁移更自由，但角色身份可能变弱。
- `0.8`：可能在身份和自然度之间更平衡。
- `1.0`：当前默认，身份强。
- `1.2`：身份更强，但可能更像训练集截图，现实/3D 迁移更困难。

### 6.2 L100 checkpoint 对比

目的：

- 观察训练步数对结果的影响。
- 判断 1200 steps 是否过拟合，或中间 checkpoint 是否更适合现实/3D 场景。

推荐先只看 `800`、`1000`、`1200`：

```bash
conda activate frieren

python3 scripts/eval_prompt_bank.py \
  --config configs/experiments/train_l100_structured.yaml \
  --prompt-bank configs/prompt_banks/frieren_eval_v1.yaml \
  --lora-dir outputs/train/frieren_sdxl_lora_hd100_full_structured_v1/checkpoint-800 \
  --model-label L100_ckpt800 \
  --output-root outputs/experiments/frieren_eval_l100_checkpoints \
  --num-images-per-prompt 2

python3 scripts/eval_prompt_bank.py \
  --config configs/experiments/train_l100_structured.yaml \
  --prompt-bank configs/prompt_banks/frieren_eval_v1.yaml \
  --lora-dir outputs/train/frieren_sdxl_lora_hd100_full_structured_v1/checkpoint-1000 \
  --model-label L100_ckpt1000 \
  --output-root outputs/experiments/frieren_eval_l100_checkpoints \
  --num-images-per-prompt 2

python3 scripts/eval_prompt_bank.py \
  --config configs/experiments/train_l100_structured.yaml \
  --prompt-bank configs/prompt_banks/frieren_eval_v1.yaml \
  --lora-dir outputs/train/frieren_sdxl_lora_hd100_full_structured_v1/checkpoint-1200 \
  --model-label L100_ckpt1200 \
  --output-root outputs/experiments/frieren_eval_l100_checkpoints \
  --num-images-per-prompt 2
```

如果 1200 和 1000 差别不大，报告里可以简单写：

> 中后期 checkpoint 的视觉质量接近，说明训练在 800 steps 后已经基本收敛；本文采用最终 1200-step 权重作为主结果。

### 6.3 不建议补的实验

不建议再做：

- 新的 rank/learning rate 消融。
- 新的数据集规模消融，比如 30 vs 80。
- 新的 1024 分辨率训练。
- IP-Adapter 或 ControlNet 主实验。

原因：

- 当前实验已经能支撑核心研究问题。
- 再加训练实验会显著增加工作量，但报告收益有限。
- IP-Adapter / ControlNet 可以作为 future work，而不是当前主线。

## 7. 报告写作路线

推荐报告实验部分按以下顺序写：

### 7.1 Experimental Setup

写清：

- Base model: SDXL base 1.0
- Method: DreamBooth-style LoRA
- Training resolution: 768
- LoRA rank: 16
- Steps: 1200
- Seed: 42
- Evaluation: 7 fixed prompts, 4 seeds per prompt

### 7.2 Dataset and Caption Design

写清三组训练数据：

- L80：80 张单角色图，结构化/半结构化 caption。
- L100：L80 加 20 张复杂/多人候选图。
- L80-simple：与 L80 相同图像，但统一简单 caption。

重点解释：

> L100 用于验证“更多但更复杂的数据”是否提升泛化；L80-simple 用于验证 caption 组织方式是否影响可控性。

### 7.3 Results: Base vs LoRA

引用：

- `p01_2d_portrait_comparison.jpg`
- `p02_2d_full_body_comparison.jpg`

结论：

> LoRA 将通用白发精灵生成转化为稳定 Frieren 角色生成。

### 7.4 Results: 80 vs 100 Images

引用：

- `p04_real_train_station_comparison.jpg`
- `p05_real_cafe_comparison.jpg`
- `p07_3d_render_comparison.jpg`

结论：

> L100 在复杂背景和风格迁移上略优，没有明显损害 2D 身份一致性，因此选择 L100 为最终模型。

### 7.5 Results: Caption Comparison

引用：

- `p02_2d_full_body_comparison.jpg`
- `p03_2d_spellcasting_comparison.jpg`

结论：

> 简单 caption 足以学习身份，但结构化 caption 更利于动作、构图和场景控制。

### 7.6 Limitations

写清：

- 3D/半写实生成仍不充分。
- LoRA 同时学习了角色身份和训练集 2D 风格，因此风格迁移受限。
- 当前评价以主观视觉分析为主，缺少自动指标。
- 更强的现实场景融合可以在 future work 中尝试 IP-Adapter 或 ControlNet。

### 7.7 Final Conclusion

可以使用这段初稿：

> Experiments show that DreamBooth-style LoRA fine-tuning on a small Frieren dataset substantially improves character identity consistency over the SDXL base model. The 80-image structured dataset is already sufficient for stable 2D character generation, while adding 20 complex samples slightly improves generalization to realistic backgrounds and semi-realistic prompts without causing obvious identity degradation. Compared with simple fixed captions, structured captions provide better control over pose, composition, and scene attributes. However, the trained LoRA tends to preserve the original 2D anime style, making full 3D or semi-realistic transformation difficult. This suggests that character identity learning is effectively handled by LoRA fine-tuning, whereas stronger cross-style transfer may require additional inference-time controls such as IP-Adapter or ControlNet.

## 8. 补充实验结果

补充实验已经完成两组：

1. L100 LoRA scale sweep：`0.6 / 0.8 / 1.0 / 1.2`
2. L100 checkpoint 对比：`800 / 1000 / 1200` steps

补充实验的作用不是重新选择主线模型，而是回答两个更细的问题：

- 推理时 LoRA 权重强度如何影响身份一致性和风格迁移？
- 1200 steps 是否必要，是否存在明显过拟合？

### 8.1 L100 LoRA scale sweep

输出目录：

`outputs/experiments/frieren_eval_l100_scale_sweep/`

该实验对 L100 使用相同 prompt bank，在 `lora_scale = 0.6, 0.8, 1.0, 1.2` 下分别生成图像。每个 scale 生成 7 个 prompt，每个 prompt 4 张，共 112 张。

#### 8.1.1 2D 全身图

![L100 scale sweep p02](../outputs/experiments/frieren_eval_l100_scale_sweep/contact_sheets_labeled/p02_2d_full_body_comparison.jpg)

观察：

- `scale=0.6` 已经能生成接近 Frieren 的角色，但身份特征相对弱一些，服装和脸部更容易偏离训练集中稳定的动画截图风格。
- `scale=0.8` 在身份和画面自然度之间较平衡，背景和人物比例比较自然。
- `scale=1.0` 身份最稳定，白金长袍、发型、耳朵和整体动漫角色感都最接近主实验结果。
- `scale=1.2` 进一步增强了角色身份，但画面更容易变得“训练集截图化”，构图和色调也更受 LoRA 风格束缚。

结论：

> 对标准 2D Frieren 生成，`scale=1.0` 是最稳的默认设置；`scale=0.8` 可以作为稍微降低 LoRA 约束、提升画面自然度的备选。

#### 8.1.2 现实车站场景

![L100 scale sweep p04](../outputs/experiments/frieren_eval_l100_scale_sweep/contact_sheets_labeled/p04_real_train_station_comparison.jpg)

观察：

- `scale=0.6` 的现实车站背景比较自然，但角色身份略弱，服装和脸部不如高 scale 稳定。
- `scale=0.8` 能较好保留 Frieren 身份，同时背景仍较自然。
- `scale=1.0` 身份更稳定，是主实验中采用的默认值。
- `scale=1.2` 虽然身份强，但角色更像被强行贴入现实场景，背景融合和画面自然度没有明显提升。

结论：

> 对“2D Frieren in realistic world”，`scale=0.8` 和 `scale=1.0` 都可用。`scale=0.8` 更偏自然融合，`scale=1.0` 更偏身份稳定。报告主结果继续使用 `scale=1.0` 更便于和其他模型公平比较。

#### 8.1.3 现实咖啡馆场景

![L100 scale sweep p05](../outputs/experiments/frieren_eval_l100_scale_sweep/contact_sheets_labeled/p05_real_cafe_comparison.jpg)

观察：

- `scale=0.6` 和 `scale=0.8` 在咖啡馆背景中显得更松弛，角色姿态和桌面交互更自然一些。
- `scale=1.0` 身份强，餐桌、杯子、咖啡馆背景也较稳定。
- `scale=1.2` 在部分 seed 中构图开始偏离原本 prompt，例如人物距离、坐姿和背景关系出现更大变化。

结论：

> 在现实背景任务中，LoRA scale 过高不一定更好。较高 scale 提升身份一致性，但可能降低与真实/摄影感背景的融合自然度。

#### 8.1.4 半写实与 3D 风格

![L100 scale sweep p06](../outputs/experiments/frieren_eval_l100_scale_sweep/contact_sheets_labeled/p06_semireal_portrait_comparison.jpg)

![L100 scale sweep p07](../outputs/experiments/frieren_eval_l100_scale_sweep/contact_sheets_labeled/p07_3d_render_comparison.jpg)

观察：

- `scale=0.6` 更容易出现柔和光影和一定的 3D/半写实体积感，但角色身份相对弱，Frieren 的标准动画特征不如高 scale 稳定。
- `scale=0.8` 在 3D/半写实任务上是比较好的折中，既保留一定身份，也允许模型响应风格迁移 prompt。
- `scale=1.0` 身份稳定，但仍明显保留 2D 动画风格。
- `scale=1.2` 进一步强化 2D Frieren 身份，反而更难变成真正 3D 或半写实；部分样本还出现更强的训练集风格束缚和局部伪影。

结论：

> LoRA scale 控制了“身份绑定”和“风格自由度”之间的权衡。低 scale 更利于风格迁移，高 scale 更利于身份一致性。由于本报告主要目标是稳定生成 Frieren，主结果仍采用 `scale=1.0`；若专门展示 3D/半写实探索，可以额外展示 `scale=0.8` 的结果。

### 8.2 L100 checkpoint 对比

输出目录：

`outputs/experiments/frieren_eval_l100_checkpoints/`

该实验比较 L100 在 `checkpoint-800`、`checkpoint-1000` 和 `checkpoint-1200` 下的生成效果。每个 checkpoint 生成 7 个 prompt，每个 prompt 2 张。

#### 8.2.1 2D 头像

![L100 checkpoint p01](../outputs/experiments/frieren_eval_l100_checkpoints/contact_sheets/p01_2d_portrait_comparison.jpg)

观察：

- `checkpoint-800` 已经能够生成稳定 Frieren 身份，说明模型在 800 steps 时已经基本学会角色。
- `checkpoint-1000` 和 `checkpoint-1200` 的脸部、发型、服装更加稳定。
- `checkpoint-1200` 的身份最接近最终主实验图，整体更干净。

结论：

> L100 在 800 steps 后已经基本收敛；继续训练到 1200 steps 主要带来身份细节和稳定性的提升。

#### 8.2.2 现实车站与咖啡馆

![L100 checkpoint p04](../outputs/experiments/frieren_eval_l100_checkpoints/contact_sheets/p04_real_train_station_comparison.jpg)

![L100 checkpoint p05](../outputs/experiments/frieren_eval_l100_checkpoints/contact_sheets/p05_real_cafe_comparison.jpg)

观察：

- 三个 checkpoint 都能完成现实背景中的 2D Frieren 生成。
- `checkpoint-800` 背景和构图已经可用，但人物身份和服装细节略弱。
- `checkpoint-1000` 在部分样本中姿态更有变化，但也可能带来更不稳定的构图。
- `checkpoint-1200` 的角色身份最稳，桌面、车站背景和服装细节也较清晰。

结论：

> 对现实背景任务，800 到 1200 steps 之间没有出现明显过拟合。1200-step 权重在身份一致性上更稳，因此适合作为最终模型。

#### 8.2.3 3D 风格

![L100 checkpoint p07](../outputs/experiments/frieren_eval_l100_checkpoints/contact_sheets/p07_3d_render_comparison.jpg)

观察：

- `checkpoint-800` 的 3D/体积感相对更自由，但身份和服装稳定性略弱。
- `checkpoint-1000` 和 `checkpoint-1200` 角色身份更强，同时也更倾向保留 2D 动画风格。
- `checkpoint-1200` 没有明显画质崩坏或过拟合，但确实更强化了训练集的 2D 风格。

结论：

> 随训练步数增加，LoRA 对 Frieren 身份和 2D 风格的绑定增强。对于本项目的主目标，这是有利的；但对于彻底 3D 化或半写实化，较早 checkpoint 或较低 LoRA scale 可能拥有更大的风格自由度。

## 9. 补充实验后的最终判断

综合主实验和补充实验，最终建议保持：

- 最佳 LoRA：`L100_structured`
- 主结果 checkpoint：`1200 steps`
- 主结果 LoRA scale：`1.0`

理由：

- `checkpoint-1200` 没有表现出明显过拟合，角色身份最稳。
- `scale=1.0` 在身份一致性和 prompt 遵循之间最均衡，适合作为报告主结果。
- `scale=0.8` 可以作为风格迁移展示的辅助结果，尤其用于现实背景或 3D/半写实探索。
- `scale=1.2` 不建议作为默认值，因为它会进一步强化训练集 2D 风格，降低现实/3D 任务中的自然度。

可以写进报告的补充结论：

> Additional inference-time analysis shows that LoRA scale provides a simple control knob between identity preservation and style flexibility. Lower scales such as 0.8 allow slightly more natural integration into realistic or 3D-like scenes, while higher scales strengthen character identity but also reinforce the original 2D anime style. Checkpoint comparison shows that the L100 LoRA has largely converged by 800 steps, and the final 1200-step checkpoint improves identity stability without obvious overfitting. Therefore, the final experiments use the 1200-step L100 LoRA with scale 1.0 as the default setting.
