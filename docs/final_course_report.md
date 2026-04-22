# 基于 SDXL LoRA 的 Frieren 角色个性化生成实验报告

姓名：  王瑞天(同组：黄超逸)
学号：  52303191027
日期：2026 年 4 月 23 日

## 摘要

本项目围绕动漫角色 Frieren 构建了一个小规模角色个性化图像生成实验。项目基于 Stable Diffusion XL base 1.0，采用 DreamBooth-style LoRA 进行轻量微调，目标是在约 100 张角色图像上学习稳定的 Frieren 身份，并进一步探索该角色在标准 2D 动漫图、现实摄影感背景以及 3D/半写实风格中的生成表现。实验比较了原始 SDXL、80 张单角色数据训练得到的 LoRA、100 张完整数据训练得到的 LoRA，以及简单 caption 版本的 LoRA。结果表明，LoRA 微调能够显著提升 Frieren 的角色身份一致性；80 张干净单角色图已经足以学习稳定身份，加入 20 张复杂/多人候选图后没有明显破坏角色一致性，并在现实背景和复杂场景泛化上略有提升；结构化 caption 相比简单 caption 更有利于动作、构图和场景控制。补充实验显示，LoRA scale 在身份保持和风格自由度之间形成可调权衡，`scale=1.0` 适合作为主结果设置，而 `scale=0.8` 在现实背景和 3D/半写实探索中更具灵活性。最终本文选择 100 张数据训练得到的 L100 LoRA，使用 1200-step checkpoint 和 `lora_scale=1.0` 作为最佳模型。

## 1. 引言

近年来，文本到图像扩散模型在开放域图像生成中取得了显著效果。然而，对于特定角色的稳定生成，仅依赖基础模型和 prompt 往往难以保持角色身份一致性。尤其是动漫角色生成任务中，模型不仅需要理解文本描述，还需要稳定复现角色的发型、服装、脸部风格、标志性配饰和整体气质。

本课程项目选择动漫角色 Frieren 作为研究对象，目标是使用小规模自建数据集微调扩散模型，使模型能够根据触发词稳定生成 Frieren，并测试该角色在不同风格和场景中的泛化能力。相比单纯生成标准动漫头像，本项目更关注三个递进任务：

- 生成标准 2D Frieren 角色图。
- 生成 2D Frieren 进入现实摄影感场景的效果。
- 尝试生成 3D 或半写实风格 Frieren。

这三个任务的难度逐渐增加。标准 2D 角色生成主要考察身份学习；现实背景混合需要模型同时保持角色身份和真实场景结构；3D/半写实生成则要求模型在强风格迁移下仍保留角色特征。因此，本项目不仅验证 LoRA 微调的有效性，也尝试分析数据划分、caption 组织和推理阶段 LoRA 强度对最终结果的影响。

## 2. 任务定义与研究问题

本项目的核心任务是：给定少量 Frieren 图像，训练一个可复用的角色 LoRA，使 SDXL 能够根据触发词 `sks_frieren` 稳定生成 Frieren。

围绕该任务，本文主要回答以下问题：

| 编号 | 研究问题 |
| --- | --- |
| RQ1 | 原始 SDXL 仅依赖 prompt 时，是否能够稳定生成 Frieren？ |
| RQ2 | LoRA 微调是否能显著提升角色身份一致性？ |
| RQ3 | 80 张干净单角色图和 100 张完整数据相比，哪种更适合作为最终训练集？ |
| RQ4 | 结构化 caption 和简单固定 caption 对可控生成有何影响？ |
| RQ5 | 在现实背景和 3D/半写实风格中，LoRA 的身份保持和风格迁移之间存在怎样的权衡？ |

## 3. 数据集构建

### 3.1 数据来源与划分

本项目共整理出 100 张 Frieren 相关图像，其中包含 80 张单角色图和 20 张复杂/多人候选图。训练集中所有图像统一转换为 RGB PNG，并以 Diffusers ImageFolder 所需的 `metadata.jsonl` 格式组织。

| 数据集 | 图像数量 | 内容 | 用途 |
| --- | ---: | --- | --- |
| `frieren_hd_single80_v1` | 80 | 单角色 Frieren 图像 | 主体身份学习 |
| `frieren_hd_all100_v1` | 100 | 80 张单角色图 + 20 张复杂/多人候选图 | 最终候选模型训练 |
| `frieren_hd_single80_simple_caption_v1` | 80 | 与 single80 相同图像，但使用简单固定 caption | Caption 对比实验 |

其中，80 张单角色图用于学习最稳定的角色身份；额外 20 张复杂/多人候选图用于探索更复杂构图和场景是否有助于泛化。该设计并不假设“更多数据一定更好”，而是用于观察额外复杂样本是否会提升现实场景和 3D/半写实场景中的表现，或反过来引入身份污染和构图混乱。

### 3.2 Caption 设计

结构化 caption 采用“触发词 + 稳定身份特征 + 当前图像动态信息”的形式。例如：

```text
sks_frieren, elf girl, long silver hair, pointy ears, white and gold robe, upper body portrait, front view, calm expression, stone wall background
```

简单 caption 则统一为：

```text
sks_frieren, elf girl, long silver hair, pointy ears, white and gold robe
```

结构化 caption 的目标是将角色身份、动作、构图和场景信息分开建模，使模型在推理时更容易响应 prompt 中的动作和场景描述。简单 caption 的目标是测试如果只将所有图像绑定到同一个身份描述，模型是否能更强地学习触发词和角色身份之间的关系。

## 4. 方法

### 4.1 基础模型

本项目使用 Stable Diffusion XL base 1.0 作为基础文生图模型，并使用 `madebyollin/sdxl-vae-fp16-fix` 作为 VAE。选择 SDXL 的原因是其图像质量较高，且 Diffusers 官方训练脚本对 DreamBooth-style LoRA 支持较成熟，适合课程项目复现。

### 4.2 DreamBooth-style LoRA

LoRA 通过在原模型部分线性层中加入低秩可训练矩阵，避免全量微调整个扩散模型，从而降低训练成本。对于本项目的小规模角色个性化任务，LoRA 有三个优势：

- 训练成本低，可以在有限计算资源下完成多组实验。
- 适合学习角色身份，不需要从零训练模型。
- 推理时可以通过 LoRA scale 调节角色身份强度。

本项目使用统一触发词 `sks_frieren`。训练阶段不重点强化 `photorealistic` 或 `3d` 等风格词，而是优先让模型学习 Frieren 身份，跨风格生成主要通过推理阶段 prompt 和 LoRA scale 进行探索。

### 4.3 训练配置

三组 LoRA 使用相同训练超参。

| 参数 | 值 |
| --- | --- |
| Base model | SDXL base 1.0 |
| Training framework | Hugging Face Diffusers |
| Method | DreamBooth-style LoRA |
| Resolution | 768 |
| LoRA rank | 16 |
| Batch size | 1 |
| Gradient accumulation | 4 |
| Learning rate | 0.0001 |
| Scheduler | constant |
| Steps | 1200 |
| Mixed precision | fp16 |
| Train text encoder | false |
| Seed | 42 |

## 5. 实验设计

### 5.1 模型组

实验共比较四组模型。

| ID | 模型 | 数据 | Caption | 作用 |
| --- | --- | --- | --- | --- |
| B0 | Base SDXL | 无 | 无 | 无微调 baseline |
| L80 | 80-image structured LoRA | 80 张单角色图 | 结构化/半结构化 caption | 已有主模型 |
| L100 | 100-image structured LoRA | 80 单角色 + 20 复杂候选 | 继承结构化 caption | 最终候选最佳模型 |
| L80-simple | 80-image simple-caption LoRA | 80 张单角色图 | 简单固定 caption | Caption 对比 |

### 5.2 统一评测 Prompt

所有模型使用同一套评测 prompt，保证对比公平。每个模型生成 7 个 prompt，每个 prompt 4 个 seed，共 28 张图。评测 prompt 分为三类：

| 类别 | Prompt ID | 目标 |
| --- | --- | --- |
| 2D Frieren | `p01_2d_portrait`, `p02_2d_full_body`, `p03_2d_spellcasting` | 标准动漫角色身份、构图和动作 |
| 2D in realistic world | `p04_real_train_station`, `p05_real_cafe` | 现实背景中的 2D 角色 |
| 3D / semi-realistic | `p06_semireal_portrait`, `p07_3d_render` | 半写实和 3D 风格迁移 |

统一推理参数如下：

| 参数 | 值 |
| --- | --- |
| Resolution | 768 x 768 |
| Inference steps | 30 |
| Guidance scale | 7.5 |
| Samples per prompt | 4 |
| Seed base | 42 |
| Default LoRA scale | 1.0 |

### 5.3 评价方式

本项目主要采用主观视觉评价，重点观察以下维度：

| 维度 | 含义 |
| --- | --- |
| Identity Consistency | 是否稳定表现 Frieren 的银发、精灵耳、白金长袍和动画气质 |
| Prompt Following | 是否遵循动作、场景、构图和风格要求 |
| Image Quality | 脸部、手部、身体结构、清晰度和伪影情况 |
| Style Transfer | 是否能进入现实背景、半写实或 3D 风格 |
| Failure Rate | 明显身份漂移、构图崩坏、多人污染或严重伪影的比例 |

由于本项目更偏课程实践和视觉分析，未引入 CLIP/DINO 等自动指标。该限制会在后文讨论。

## 6. 实验结果与分析

### 6.1 训练收敛情况

三组 LoRA 均正常收敛，final loss 接近。

| 模型 | Initial loss | Final loss | Min loss | 观察 |
| --- | ---: | ---: | ---: | --- |
| L80 | 0.153391 | 0.011553 | 0.001407 | 正常收敛 |
| L100 | 0.141887 | 0.011291 | 0.001372 | 正常收敛，final loss 略低 |
| L80-simple | 0.154425 | 0.011773 | 0.001431 | 正常收敛，final loss 略高 |

Loss 只能说明训练过程稳定，不能直接代表视觉质量。最终模型选择仍主要依据统一评测结果。

### 6.2 原始 SDXL 与 LoRA 对比

![标准 2D 头像对比](../outputs/experiments/frieren_eval_v1/contact_sheets/p01_2d_portrait_comparison.jpg)

图中可以看到，B0 能生成白发精灵女性，但结果更接近通用 fantasy elf，而不是稳定 Frieren。相比之下，L80、L100 和 L80-simple 都显著强化了 Frieren 身份，包括银白长发、精灵耳、白金长袍、红色耳坠以及更接近动画截图的脸部风格。

![2D 全身图对比](../outputs/experiments/frieren_eval_v1/contact_sheets/p02_2d_full_body_comparison.jpg)

在全身图中，B0 的画面更偏写实 fantasy illustration，角色身份仍然泛化。L80 与 L100 能稳定生成接近 Frieren 的动画角色，并保留白金长袍。L80-simple 也学到了身份，但更容易出现背面或侧后方视角，说明简单 caption 对构图和视角的约束较弱。

结论：

> 原始 SDXL 具备生成白发精灵和 fantasy 场景的能力，但缺乏特定角色身份知识。LoRA 微调显著提升了 Frieren 的身份一致性，说明小规模角色数据集足以完成角色个性化生成。

### 6.3 80 张与 100 张数据对比

![现实车站对比](../outputs/experiments/frieren_eval_v1/contact_sheets/p04_real_train_station_comparison.jpg)

在现实车站场景中，B0 能生成较自然的站台背景，但角色偏写实白发精灵。L80 和 L100 都能生成“2D 动漫角色 + 现实车站背景”的混合效果。其中 L100 在背景透视、站台元素和人物融合上略自然，没有明显因为加入 20 张复杂样本而产生多人污染。

![现实咖啡馆对比](../outputs/experiments/frieren_eval_v1/contact_sheets/p05_real_cafe_comparison.jpg)

在咖啡馆场景中，L80 和 L100 都较好实现了 2D 角色进入现实/摄影感背景的效果。L100 的餐桌、杯子和咖啡馆光照更稳定一些，说明复杂样本可能对现实背景泛化有轻微帮助。

结论：

> 80 张干净单角色图已经足以学习稳定 Frieren 身份。加入额外 20 张复杂/多人候选图后，L100 没有明显损害角色一致性，并在现实背景和复杂场景泛化上表现出轻微优势。因此本文选择 L100 作为最终最佳 LoRA。

### 6.4 Caption 组织方式对比

![施法动作对比](../outputs/experiments/frieren_eval_v1/contact_sheets/p03_2d_spellcasting_comparison.jpg)

L80-simple 使用与 L80 相同的 80 张图，但所有图像都使用简单固定 caption。实验发现，L80-simple 能够学习 Frieren 身份，但在全身图、施法动作和现实背景中更容易出现视角偏差、背面构图或动作表达不足。结构化 caption 则能够将动作、构图和场景信息更明确地提供给模型，使推理时的 prompt 控制更稳定。

结论：

> 简单 caption 可以让触发词绑定角色身份，但结构化 caption 对构图、动作和场景控制更有帮助。对于本项目这类需要比较 2D、现实背景和 3D prompt 的任务，结构化 caption 更适合作为主方案。

### 6.5 3D 与半写实生成分析

![半写实对比](../outputs/experiments/frieren_eval_v1/contact_sheets/p06_semireal_portrait_comparison.jpg)

![3D 风格对比](../outputs/experiments/frieren_eval_v1/contact_sheets/p07_3d_render_comparison.jpg)

B0 更容易响应半写实或 3D 风格 prompt，但角色身份弱。LoRA 模型能够保持 Frieren 身份，但往往仍保留明显的 2D 动画截图风格。尤其在 `3d stylized character render` prompt 下，L80 和 L100 都没有完全转化为 3D 角色，而是更像带有少量体积光影的动画角色。

这说明 LoRA 学到的不仅是角色身份，也包括训练数据中的 2D 动画风格。当 LoRA 强度较高时，角色身份更稳定，但跨风格迁移更困难。

结论：

> 3D/半写实生成是当前方案的主要短板。LoRA 对角色身份绑定很强，但也把训练集中 2D 动画风格一起绑定进模型，导致 prompt 很难完全把角色推向 3D 或半写实。

### 6.6 主观评分汇总

下表为基于统一评测图像的主观视觉评分，分数范围为 1 到 5。该表用于概括趋势，不代表严格自动指标。

| 模型 | 身份一致性 | Prompt 遵循 | 图像质量 | 现实背景混合 | 3D/半写实迁移 | 综合观察 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B0 | 2.0 | 4.0 | 4.0 | 3.5 | 3.5 | 场景和风格响应较好，但不是稳定 Frieren |
| L80 | 4.8 | 4.0 | 4.1 | 4.0 | 2.6 | 2D 身份很稳，风格迁移受限 |
| L100 | 4.9 | 4.2 | 4.2 | 4.3 | 2.9 | 综合最好，现实背景略优 |
| L80-simple | 4.4 | 3.5 | 3.8 | 3.8 | 2.5 | 身份可学到，但构图和动作控制较弱 |

## 7. 补充实验

### 7.1 LoRA Scale Sweep

为了分析推理阶段 LoRA 强度的影响，本文对 L100 测试了 `lora_scale = 0.6, 0.8, 1.0, 1.2`。

![L100 scale sweep 全身图](../outputs/experiments/frieren_eval_l100_scale_sweep/contact_sheets_labeled/p02_2d_full_body_comparison.jpg)

在标准 2D 全身图中，`scale=1.0` 的 Frieren 身份最稳定，服装、发型和整体动画角色感都最接近主实验结果。`scale=0.8` 在身份和画面自然度之间较平衡。`scale=1.2` 进一步增强了角色身份，但画面更容易变得训练集截图化。

![L100 scale sweep 现实车站](../outputs/experiments/frieren_eval_l100_scale_sweep/contact_sheets_labeled/p04_real_train_station_comparison.jpg)

在现实车站中，`scale=0.8` 和 `scale=1.0` 都可用。`scale=0.8` 更偏自然融合，`scale=1.0` 更偏身份稳定。`scale=1.2` 并没有明显提升现实场景融合，反而更强化 2D 角色贴入背景的感觉。

![L100 scale sweep 3D](../outputs/experiments/frieren_eval_l100_scale_sweep/contact_sheets_labeled/p07_3d_render_comparison.jpg)

在 3D prompt 中，较低 scale 更容易产生柔和光影和体积感，但身份会变弱；较高 scale 能保留身份，但更难真正 3D 化。

结论：

> LoRA scale 控制了身份绑定和风格自由度之间的权衡。低 scale 更利于风格迁移，高 scale 更利于身份一致性。由于本文主目标是稳定生成 Frieren，主结果采用 `scale=1.0`；若专门展示 3D 或半写实探索，可以额外展示 `scale=0.8`。

### 7.2 Checkpoint 对比

本文还比较了 L100 的 `checkpoint-800`、`checkpoint-1000` 和 `checkpoint-1200`。

![L100 checkpoint 头像](../outputs/experiments/frieren_eval_l100_checkpoints/contact_sheets/p01_2d_portrait_comparison.jpg)

800 steps 时模型已经基本学会 Frieren 身份；1000 和 1200 steps 的脸部、发型和服装更加稳定。

![L100 checkpoint 现实车站](../outputs/experiments/frieren_eval_l100_checkpoints/contact_sheets/p04_real_train_station_comparison.jpg)

现实背景任务中，三个 checkpoint 都能完成基本生成。1200-step 权重的人物身份最稳，背景和服装细节也较清晰。

![L100 checkpoint 3D](../outputs/experiments/frieren_eval_l100_checkpoints/contact_sheets/p07_3d_render_comparison.jpg)

3D 任务中，800-step checkpoint 的风格自由度略高，但身份和服装稳定性较弱；1200-step checkpoint 更稳定，也更强化 2D 动画风格。

结论：

> L100 在 800 steps 后已经基本收敛。继续训练到 1200 steps 没有出现明显过拟合，并提升了角色身份稳定性。因此最终采用 1200-step checkpoint。

## 8. 最终模型选择

综合主实验和补充实验，本文最终选择：

| 项目 | 选择 |
| --- | --- |
| 最佳 LoRA | L100 structured LoRA |
| 权重路径 | `outputs/train/frieren_sdxl_lora_hd100_full_structured_v1` |
| Checkpoint | 1200 steps |
| Default LoRA scale | 1.0 |

选择理由：

- L100 相较 B0 显著提升 Frieren 身份一致性。
- L100 相较 L80 在标准 2D 任务中没有明显退化。
- L100 在现实车站、咖啡馆和部分半写实/3D prompt 上略有更好的复杂场景泛化。
- 1200-step checkpoint 没有明显过拟合，身份最稳。
- `scale=1.0` 在身份一致性和 prompt 遵循之间最均衡，适合作为主结果。

## 9. 局限性

本项目仍存在以下局限：

| 局限 | 说明 |
| --- | --- |
| 评价方式较主观 | 当前主要依赖视觉对比，没有引入 CLIP、DINO 或人类多评审打分 |
| 数据规模仍较小 | 100 张图足以完成课程项目，但难以覆盖所有姿态、表情、光照和场景 |
| 3D/半写实迁移不足 | LoRA 同时学习了角色身份和 2D 动画风格，导致强风格迁移困难 |
| 现实背景融合有限 | 角色与现实背景有时像“贴入”画面，而不是真正统一渲染 |
| 未使用推理增强 | 本文没有正式引入 IP-Adapter、ControlNet 或参考图引导 |

后续可以从以下方向改进：

- 使用 IP-Adapter 引入现实场景参考图，加强光影和背景一致性。
- 使用 ControlNet 控制姿态、构图或深度结构。
- 引入更多半写实或 3D 风格 Frieren 数据，单独训练风格迁移版本。
- 使用 CLIP 图文一致性、DINO 图像相似度和人工多评审评分进行更系统评价。

## 10. 结论

本文完成了一个围绕 Frieren 角色的 SDXL LoRA 个性化生成项目。实验表明，原始 SDXL 能够生成白发精灵和 fantasy 场景，但无法稳定生成 Frieren；DreamBooth-style LoRA 微调能够显著提升角色身份一致性，使模型稳定生成银白长发、精灵耳、白金长袍和动画风格脸部特征。

在数据划分方面，80 张干净单角色图已经足以学习稳定 2D Frieren；加入额外 20 张复杂/多人候选图后，L100 没有明显身份退化，并在现实背景和复杂场景泛化上略有优势，因此被选为最终最佳模型。在 caption 设计方面，简单 caption 能够绑定角色身份，但结构化 caption 对动作、构图和场景控制更有帮助。

补充实验进一步显示，LoRA scale 是身份保持和风格自由度之间的控制旋钮。`scale=1.0` 适合作为稳定角色生成的默认设置，`scale=0.8` 更适合探索现实背景和 3D/半写实风格。Checkpoint 对比显示 L100 在 800 steps 后已经基本收敛，1200-step checkpoint 在身份稳定性上最优且没有明显过拟合。

总体而言，本项目说明了小规模角色数据集配合 SDXL LoRA 可以有效完成动漫角色个性化生成；但若希望进一步实现高质量 3D/半写实迁移或更自然的现实场景融合，仅依赖 prompt 和角色 LoRA 仍然有限，后续需要结合 IP-Adapter、ControlNet 或更有针对性的跨风格数据。

## 附录 A：核心 Prompt 示例

```text
sks_frieren, elf girl, long silver hair, pointy ears, white and gold robe, upper body portrait, calm expression, soft daylight, detailed anime illustration
```

```text
sks_frieren, 2d anime elf girl, long silver hair, pointy ears, white and gold robe, standing in a realistic modern train station, photographic lighting, realistic background, anime character rendering
```

```text
sks_frieren, 3d stylized elf girl, long silver hair, pointy ears, white and gold robe, cinematic lighting, high quality 3d character render
```

## 附录 B：主要输出路径

| 内容 | 路径 |
| --- | --- |
| 最终实验规划 | `docs/final_experiment_plan.md` |
| 实验分析底稿 | `docs/experiment_analysis.md` |
| 最终课程报告 | `docs/final_course_report.md` |
| Prompt bank | `configs/prompt_banks/frieren_eval_v1.yaml` |
| 主评测结果 | `outputs/experiments/frieren_eval_v1/` |
| Scale sweep 结果 | `outputs/experiments/frieren_eval_l100_scale_sweep/` |
| Checkpoint 对比结果 | `outputs/experiments/frieren_eval_l100_checkpoints/` |
| 最佳 LoRA | `outputs/train/frieren_sdxl_lora_hd100_full_structured_v1/` |

