<h1 align="center">[ICCV 2025] LUT-Fuse</h1>
<p align="center">
  <em>Towards Extremely Fast Infrared and Visible Image Fusion via Distillation to Learnable Look-Up Tables</em>
</p>

<p align="center">
  <a href="https://github.com/zyb5/LUT-Fuse" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/GitHub-Code-black?logo=github" alt="Code" />
  </a>
  <a href="https://arxiv.org/abs/2509.00346" style="text-decoration:none; margin-left:8px;">
    <img src="https://img.shields.io/badge/arXiv-Paper-B31B1B?logo=arxiv" alt="Paper" />
  </a>
  <a href="https://huggingface.co/spaces/ZYB5/LUT-Fuse-demo" style="text-decoration:none; margin-left:8px;">
    <img src="https://img.shields.io/badge/HuggingFace%20-Demo-ffcc00?logo=huggingface" alt="Hugging Face" />
  </a>
</p>


<p align="center">
  <img src="assets/framework.png" alt="LUT-Fuse Framework" width="90%">
</p>

---

## ⚙️ Environment

```
conda create -n lutfuse python=3.8
conda activate lutfuse
```

```
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## 📂 Dataset

You should list your dataset as followed rule:

```
|dataset
  |train
    |Infrared
    |Visible
    |Fuse_ref
  |test
    |Infrared
    |Visible
    |Fuse_ref
```

## 💾 Checkpoints

We provide our **pretrained checkpoints** directly in this repository for convenience.  
You can find them under [`./ckpts`](./ckpts).

- **Fusion LUT weights:** `ckpts/fine_tuned_lut.npy`  
- **Context generator weights:** `ckpts/generator_context.pth`

## 🧪 Test

```bash
CUDA_VISIBLE_DEVICES=0 python test_lut.py \
  --visible_dir /path/to/Visible \
  --infrared_dir /path/to/Infrared \
  --save_dir /path/to/results
```

The default checkpoints are `ckpts/fine_tuned_lut.npy` and
`ckpts/generator_context.pth`. Override them with `--lut_path` and
`--context_path` when needed.

The test script performs one unmeasured warm-up pass, then reports total
inference time, average latency, FPS, input image size, parameter/LUT counts,
and peak CUDA memory. The same summary is saved to
`benchmark_metrics.txt` in `--save_dir`.

## 🚀 Train

```
CUDA_VISIBLE_DEVICES=0 python fine_tune_lut.py
```

### A0/A1 teacher distillation

`train_distillation.py` keeps the original LUT, context encoder, interpolation,
and four original training terms. Only the teacher target is configurable:

- `A0_original`: cached RGB images produced by the original MM-Net. The
  upstream repository does not include an MM-Net class/checkpoint, so the
  existing `Fuse_ref`-style images are the baseline teacher contract.
- `A1_lefuse_teacher`: a frozen LEFuse teacher evaluated online under
  `torch.no_grad()`. LEFuse is training-only and is never imported by
  `test_lut.py` or `scripts/calculate.py`.

Edit the dataset paths in the selected config, then run:

```bash
# Original cached-teacher baseline
CUDA_VISIBLE_DEVICES=0 python train_distillation.py --config configs/a0_original.yaml

# Teacher-replacement-only experiment
CUDA_VISIBLE_DEVICES=0 python train_distillation.py --config configs/a1_lefuse_teacher.yaml
```

The default A1 config expects the ETRI LEFuse checkout at
`../ETRI_Night_Fusion` and checkpoint `../ETRI_Night_Fusion/L2024.pth`.
Change `teacher.source_dir` and `teacher.checkpoint` when using another
location. LEFuse preprocessing is reproduced from that checkout, including
its fixed nighttime luminance/chroma processing. Because its OpenCV NLM step
accepts one image at a time, online teacher generation iterates over each
training batch internally; this does not affect deployed inference.

Output contracts used for distillation are:

- LEFuse teacher: reconstructed RGB float tensor `[B, 3, H, W]`, clamped to
  `[0, 1]`.
- LUT-Fuse student: original reconstructed RGB float tensor `[B, 3, H, W]`.
  It remains unclamped to preserve upstream training/inference behavior. The
  LUT checkpoint itself is unconstrained, so the realized RGB values may
  leave `[0, 1]`; the trainer reports this range and rejects NaN/Inf.
- Comparison: direct, spatially aligned RGB L1. No resize or channel broadcast
  is performed between teacher and student.

At the first training batch the trainer asserts and prints all input/output
shapes and ranges. Every run writes its resolved YAML, seed, git commit,
teacher/student output contract, TensorBoard losses, and best/final student
checkpoints under `finetune_lut_exp/`.

A1 intentionally adds no texture, enhancement, perceptual, weighted-KD, or
Multi-LUT terms. Before proceeding to A2, compare A1 against A0 using the same
data and preprocessing. Check whether dark visible structure and texture
improve while thermal targets and color remain stable. Benchmark the saved A1
student independently with `test_lut.py`; the inference graph should retain
the original LUT-Fuse speed because no teacher component is present.

## 📖 Citation

If you find our work or dataset useful for your research, please cite our paper.

```bibtex
@inproceedings{yi2025LUT-Fuse,
  title={LUT-Fuse: Towards Extremely Fast Infrared and Visible Image Fusion via Distillation to Learnable Look-Up Tables},
  author={Yi, Xunpeng and Zhang, Yibing and Xiang, Xinyu and Yan, Qinglong and Xu, Han and Ma, Jiayi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```

If you have any questions, please send an email to zhangyibing@whu.edu.cn


