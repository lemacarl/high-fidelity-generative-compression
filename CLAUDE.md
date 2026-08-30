# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch implementation of "High-Fidelity Generative Image Compression" (Mentzer et al., arXiv:2006.09965). The model achieves 100x+ compression ratios with perceptually-pleasing reconstructions using a GAN-based decoder conditioned on hierarchical entropy-coded latents. **Reconstructions may synthesize image details** — not suitable for medical images or documents.

## Common Commands

### Training

```bash
# Stage 1: Base rate-distortion model
python3 train.py --model_type compression --regime low --n_steps 1e6

# Stage 2: GAN fine-tuning (warmstart from Stage 1 checkpoint)
python3 train.py --model_type compression_gan --regime low --n_steps 2e5 \
  --warmstart --ckpt path/to/base/checkpoint
```

Regimes target different bitrates: `low` (~0.14 bpp), `med` (~0.30 bpp), `high` (~0.45 bpp).

Memory-saving flags if OOM: `--batch_size 4`, `--latent_channels 128`, `--n_residual_blocks 5`, `--crop_size 192`.

### Compression

```bash
# Reconstruct images (no entropy coding)
python3 compress.py -i path/to/images -ckpt path/to/checkpoint --reconstruct

# Compress to .hfc format (with entropy coding)
python3 compress.py -i path/to/images -ckpt path/to/checkpoint --save
```

### Monitoring

```bash
tensorboard --logdir experiments/<name>/tensorboard --port 2401
```

### Verify Setup

```bash
python3 -m src.model
```

## Architecture

The pipeline is: **Image → Encoder → Quantized Latents y → Generator → Reconstruction**, with a parallel **Hyperprior** entropy model over y and hyperlatents z for compression.

### Key Components

| Module | Role |
|--------|------|
| `src/model.py` | Orchestrates all components; entry point for train/compress/decompress |
| `src/network/encoder.py` | 4-layer conv downsampler: RGB → (C=220, H/16, W/16) latents |
| `src/network/generator.py` | 4-layer transposed conv upsampler with 9 residual blocks |
| `src/network/hyper.py` | Hyperprior analysis (y→z) and synthesis (z→params for y distribution) nets |
| `src/hyperprior.py` | Wraps hyper networks; computes rate (bits) for latents y and hyperlatents z |
| `src/network/discriminator.py` | PatchGAN discriminator; takes `[reconstruction, encoder_output]` as input |
| `src/loss/losses.py` | Rate-distortion loss + optional GAN + LPIPS perceptual loss |
| `src/compression/hyperprior_model.py` | Entropy coding wrapper using vectorized ANS |
| `src/compression/entropy_coding.py` | Vectorized ANS implementation |
| `src/compression/compression_utils.py` | `.hfc` binary format I/O |
| `src/helpers/datasets.py` | OpenImages and CityScapes data loaders |
| `default_config.py` | All hyperparameters and model configurations |

### Loss Function

```
Loss = λ_A * R(y) + k_M * D(x, x̂) + k_P * L_LPIPS(x, x̂) + β * L_GAN
```

- Rate R(y) uses a two-level hierarchical entropy model (latents y conditioned on hyperlatents z)
- Two model types: `ModelTypes.COMPRESSION` (no GAN) and `ModelTypes.COMPRESSION_GAN`
- Training uses additive uniform noise for quantization; inference uses hard rounding

### Entropy Coding

Compression produces `.hfc` binary files. The process:
1. Quantize hyperlatents z → entropy code z
2. Use hyperprior synthesis net to get probability parameters for y given z
3. Entropy code y using those parameters (vectorized ANS)
4. Pack both coded streams + spatial metadata into `.hfc`

ANS coding is CPU-bound; GPU strongly recommended for encoder/decoder but ANS runs on CPU.

### Normalization Options

- Channel normalization (default): `use_channel_norm=True`
- Instance normalization: `use_channel_norm=False`

## Configuration

All hyperparameters live in `default_config.py`. Key flags to `train.py`:

- `--model_type`: `compression` or `compression_gan`
- `--regime`: `low`, `med`, or `high`
- `--likelihood_type`: `gaussian` (default) or `logistic`
- `--use_channel_norm` / `-norm`: enable channel normalization
- `--warmstart --ckpt <path>`: resume from checkpoint for GAN fine-tuning

Checkpoints are saved under `experiments/<timestamp>/checkpoints/`.

## TFLite Refactor (`tflite/`)

A complete parallel reimplementation in TensorFlow/Keras targeting edge deployment (Raspberry Pi). Produces four standalone `.tflite` models (encoder, hyper_encoder, hyper_decoder, decoder) plus a NumPy-only ANS entropy coder.

### Key architectural differences from PyTorch path

| Aspect | PyTorch (`src/`) | TFLite (`tflite/`) |
|--------|-----------------|-------------------|
| Encoder | Custom 4-layer conv | MobileNetV3Small backbone + 1×1 projection head |
| Latent channels | 220 | 96 |
| Hyper channels | 128 | 192 |
| Upsampling | ConvTranspose2D | UpSampling2D (bilinear) + Conv2D (avoids INT8 checkerboard) |
| Padding | ReflectionPad2d | `"same"` padding |
| Normalization | ChannelNorm/InstanceNorm | LayerNormalization |
| Framework | PyTorch | TensorFlow/Keras |

The decoder matches the original HIFIC Generator (~25M params). The discriminator (`tflite/training/trainer.py`) is a full-capacity PatchGAN with SpectralNormalization; it is used only during GAN training and is never exported to TFLite.

### Dependencies

```bash
pip install -r tflite/requirements_tflite.txt   # GPU workstation
# Raspberry Pi: pip install tflite-runtime Pillow numpy scipy
```

### Training (TFLite path)

```bash
# Phase 1: compression-only
python -m tflite.training.trainer \
    --dataset_path data/openimages --regime low --n_steps 500000 \
    --checkpoint_dir experiments/tflite_low/

# Phase 2: GAN fine-tuning
python -m tflite.training.trainer \
    --dataset_path data/openimages --regime low \
    --model_type compression_gan --warmstart \
    --checkpoint experiments/tflite_low/ckpt-500000 \
    --n_steps 200000 --checkpoint_dir experiments/tflite_low_gan/
```

Training saves `density_weights.npz` (factorized prior weights for ANS) alongside checkpoints.

### Export to TFLite

```bash
# FP32 export
python -m tflite.conversion.export_tflite \
    --checkpoint experiments/tflite_low/final-500000 --out_dir tflite_models/

# FP32 + INT8 (provide calibration images)
python -m tflite.conversion.export_tflite \
    --checkpoint experiments/tflite_low/final-500000 --out_dir tflite_models/ \
    --int8 --image_dir data/train/
```

Produces `encoder.tflite`, `hyper_encoder.tflite`, `hyper_decoder.tflite`, `decoder.tflite` (and `*_int8.tflite` variants). The exporter audits for NaN/Inf in BatchNorm moving stats and sanitizes them before conversion — a known failure mode when `training=False` exposes stale BN stats.

### Evaluation (GPU, no TFLite conversion needed)

```bash
python -m tflite.evaluate \
    --checkpoint experiments/tflite_low/final-500000 \
    --images data/test_inputs/image.png --out_dir eval_out/
```

### Inference on device (Raspberry Pi)

```bash
# Compress
python -m tflite.inference.compress --compress \
    -i photo.jpg -o photo.hfc \
    --models_dir ~/compression/tflite_models/ \
    --density_weights ~/compression/density_weights.npz

# Decompress
python -m tflite.inference.compress --decompress \
    -i photo.hfc -o photo_recon.png \
    --models_dir ~/compression/tflite_models/ \
    --density_weights ~/compression/density_weights.npz
```

Compress loads only encoder+hyper_encoder+hyper_decoder; decompress loads only hyper_decoder+decoder — avoids loading unused models on memory-constrained hardware.

The `tflite/compression/` directory mirrors `src/compression/` but with all PyTorch dependencies removed. The ANS coder (`entropy_coding.py`) is pure NumPy. `FactorizedPriorNumpy` in `entropy_models.py` reconstructs the prior CDF tables from the exported `density_weights.npz` for use on the Pi.
