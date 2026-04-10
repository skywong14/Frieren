# Environment Setup And Local Model Download

本说明面向当前仓库的第一版 MVP，目标是：
- 安装最小可运行依赖
- 把 SDXL base 和 VAE 下载到本地 `/home/rtwang/models`
- 用当前的 6 张基础 PNG 先跑通一次 smoke test

## 1. 当前本机状态

我刚刚在这台机器上确认到：
- GPU: `NVIDIA H800 NVL`
- Python 环境里已经有：`torch`, `torchvision`, `transformers`, `datasets`, `Pillow`, `PyYAML`
- 还缺：`accelerate`, `diffusers`, `peft`
- `hf` CLI 已可用，并且当前账号已登录

说明：如果你继续在当前环境里操作，不需要重新装 PyTorch，只补本项目依赖即可。

## 2. 安装最小依赖

在仓库根目录执行：

```bash
cd /home/rtwang/Frieren
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

安装完成后，建议快速检查：

```bash
python3 - <<'PY'
import accelerate, diffusers, peft, transformers, datasets, PIL, yaml
print('environment ok')
PY
```

## 3. 如果你想新建一个干净环境

如果你不想污染当前 Python 环境，可以新建一个虚拟环境：

```bash
cd /home/rtwang/Frieren
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

然后安装与你 CUDA 匹配的 PyTorch，再安装项目依赖。

如果你沿用当前机器的 CUDA 12.8 路线，可以优先尝试：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

如果这条 PyTorch 安装命令在你的环境里有版本冲突，就保留当前已有的 torch，只执行：

```bash
pip install -r requirements.txt
```

## 4. 下载模型到 `/home/rtwang/models`

本项目第一版需要两个模型：
- Base model: `stabilityai/stable-diffusion-xl-base-1.0`
- VAE: `madebyollin/sdxl-vae-fp16-fix`

我已经确认这两个仓库当前都是 `gated: false`，可以直接下载。

先创建本地目录：

```bash
mkdir -p /home/rtwang/models
```

### 4.1 下载 SDXL base 1.0

```bash
hf download stabilityai/stable-diffusion-xl-base-1.0 \
  --local-dir /home/rtwang/models/stable-diffusion-xl-base-1.0
```

### 4.2 下载 SDXL fp16-fix VAE

```bash
hf download madebyollin/sdxl-vae-fp16-fix \
  --local-dir /home/rtwang/models/sdxl-vae-fp16-fix
```

### 4.3 下载完成后检查

```bash
ls /home/rtwang/models/stable-diffusion-xl-base-1.0
ls /home/rtwang/models/sdxl-vae-fp16-fix
```

你应该至少能看到：
- base 模型目录里有 `model_index.json`、`unet/`、`text_encoder/`、`text_encoder_2/`、`tokenizer/`、`scheduler/`
- VAE 目录里有 `config.json` 和 `diffusion_pytorch_model.safetensors`

## 5. 把训练配置切到本地模型路径

下载完后，把配置文件中的模型 ID 改成本地路径：

文件：`configs/train_sdxl_lora.yaml`

把这两项：

```yaml
model:
  base_model: stabilityai/stable-diffusion-xl-base-1.0
  vae_model: madebyollin/sdxl-vae-fp16-fix
```

改成：

```yaml
model:
  base_model: /home/rtwang/models/stable-diffusion-xl-base-1.0
  vae_model: /home/rtwang/models/sdxl-vae-fp16-fix
```

这样后续训练和推理都直接走本地模型，不依赖在线拉取。

## 6. 用当前 6 张图先做 smoke test

你现在的原始图在：
- `data/raw/chara1_face1.png`
- `data/raw/chara1_face2.png`
- `data/raw/chara1_face3.png`
- `data/raw/chara1_face4.png`
- `data/raw/chara1_full1.png`
- `data/raw/chara1_full2.png`

其中 4 张只有 `248x248`，2 张是 `750x683`。这套数据足够跑通流程，但不适合评价质量。

### 6.1 先把图复制到 processed

```bash
mkdir -p data/processed/images
cp data/raw/*.png data/processed/images/
```

### 6.2 生成 metadata.jsonl

```bash
python3 scripts/build_metadata.py --overwrite
```

检查结果：

```bash
sed -n '1,20p' data/processed/metadata.jsonl
```

### 6.3 第一次训练前的保守建议

为了让这 6 张小图更容易先跑通，建议第一次先临时把配置改得更保守：

```yaml
training:
  resolution: 512
  max_train_steps: 100
  checkpointing_steps: 50

validation:
  every_n_epochs: 1
```

原因：
- `248x248` 直接上 `1024` 会被严重放大
- smoke test 的目标是先确认训练链路和输出路径没问题，不是追求成图质量

正式实验再切回 `1024` 和更长训练步数。

## 7. 训练脚本路径说明

当前仓库的 `scripts/train_sdxl_lora.sh` 不会自动下载 Diffusers 官方训练脚本。

你需要二选一：

推荐先用下面这组命令把官方脚本放到仓库约定位置：

```bash
cd /home/rtwang/Frieren
git clone https://github.com/huggingface/diffusers.git third_party/diffusers
```

完成后，训练脚本默认就会去找：

```text
third_party/diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py
```

如果后面你遇到“官方脚本参数和已安装 diffusers 版本不一致”的错误，再补这一条：

```bash
python3 -m pip install -e third_party/diffusers
```

### 方案 A：通过环境变量指定脚本路径

如果你本地已经有 diffusers 仓库：

```bash
export DIFFUSERS_TRAIN_SCRIPT=/path/to/diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py
```

### 方案 B：把官方脚本放到仓库约定位置

```bash
mkdir -p third_party/diffusers/examples/dreambooth
cp /path/to/train_dreambooth_lora_sdxl.py third_party/diffusers/examples/dreambooth/
```

## 8. 先做 dry-run

在真正训练前，先确认命令展开正常：

```bash
scripts/train_sdxl_lora.sh --dry-run
```

如果 dry-run 正常，再启动训练：

```bash
scripts/train_sdxl_lora.sh
```

## 9. 训练完成后做推理对比

输出目录默认在：
- 训练：`outputs/train/frieren_sdxl_lora_v1/`
- 推理：`outputs/infer/frieren_sdxl_lora_v1/`

推理命令示例：

```bash
python3 scripts/infer_compare.py --config configs/train_sdxl_lora.yaml
```

如果想只生成 LoRA 图、不生成 base 对比：

```bash
python3 scripts/infer_compare.py --skip-base
```

## 10. 当前最推荐的执行顺序

```bash
cd /home/rtwang/Frieren
python3 -m pip install -r requirements.txt
hf download stabilityai/stable-diffusion-xl-base-1.0 --local-dir /home/rtwang/models/stable-diffusion-xl-base-1.0
hf download madebyollin/sdxl-vae-fp16-fix --local-dir /home/rtwang/models/sdxl-vae-fp16-fix
mkdir -p data/processed/images
cp data/raw/*.png data/processed/images/
python3 scripts/build_metadata.py --overwrite
scripts/train_sdxl_lora.sh --dry-run
```

如果上面都通过，再进入真实训练。
