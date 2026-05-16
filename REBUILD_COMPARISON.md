# HIFIC → TFLite Rebuild: Architecture & Design Comparison

## Overview

| Dimension | Original (HIFIC) | Rebuild (TFLite) |
|-----------|-----------------|-----------------|
| Framework | PyTorch 1.6 | TensorFlow 2.x / TFLite |
| Target runtime | GPU workstation | Raspberry Pi 4 (ARM Cortex-A72) |
| Inference format | PyTorch `.pt` checkpoint | Four `.tflite` flatbuffers |
| INT8 quantization | No | Yes (post-training, ARM-optimised) |
| Approx. model size | ~737 MB | <60 MB total (4 models combined) |

---

## Encoder

| | Original | Rebuild |
|---|---|---|
| Backbone | Custom conv stack from scratch | MobileNetV3 Small (pretrained ImageNet) |
| Architecture | 5 downsampling Conv2d blocks | MobileNetV3 feature extractor + Conv2d(96, 1) projection head |
| Channel progression | 3 → 60 → 120 → 240 → 480 → 960 | Pretrained backbone → 96 (projection) |
| Intermediate channels | 960 (bottleneck) | 96 (latent space) |
| Latent channels | 220 | 96 |
| Normalisation | ChannelNorm2D or InstanceNorm2D | BatchNorm (inside MobileNetV3) |
| Padding | ReflectionPad2d | Same-padding (TFLite compatible) |
| Downsampling | Strided Conv2d ×4 | MobileNetV3 stride schedule ×4 |
| Output spatial | 16×16 | 16×16 |
| Warm-start | Random init | ImageNet pretrained weights |

**Why:** MobileNetV3 Small provides strong feature extraction at a fraction of the compute cost. Pretrained ImageNet weights dramatically reduce the data and training steps needed for the encoder to learn useful representations. The 960-channel bottleneck of the original is replaced with 96-channel latents — small enough to fit in Pi memory, large enough to preserve quality at 0.14–0.45 bpp.

---

## Decoder (Generator)

| | Original | Rebuild |
|---|---|---|
| Architecture | 9 residual blocks + 4 upsampling layers | 3 inverted residual blocks + 4 bilinear upsamplings |
| Block type | Standard residual (two 3×3 Conv2d) | Inverted residual (depthwise-separable, expansion ×4) |
| Channel progression | 960 → 480 → 240 → 120 → 60 → 3 | 96 → 96 → 96 → 128 → 64 → 32 → 3 |
| Upsampling | ConvTranspose2d ×4 (stride=2) | UpSampling2D (bilinear) + DepthwiseConv2D ×4 |
| Final activation | Tanh (output in [-1, 1]) | Sigmoid (output in [0, 1]) |
| Normalisation | ChannelNorm or InstanceNorm | BatchNorm |
| Optional noise injection | Yes (32-dim concatenated noise) | No |

**Why:** ConvTranspose2d produces checkerboard artefacts in INT8 TFLite on ARM. Bilinear upsample + depthwise conv avoids this entirely and is natively supported by the TFLite ARM delegate. Inverted residuals (MobileNet-style) achieve similar representational capacity at 5–10× fewer multiply-accumulates than standard residual blocks.

---

## Hyperprior

| | Original | Rebuild |
|---|---|---|
| Analysis channels | 320 (large) / 192 (small) | 128 |
| Synthesis channels | 320 (large) / 192 (small) | 128 |
| Hyperlatent channels | 320 / 192 | 128 |
| Analysis upsampling | ConvTranspose2d ×2 | UpSampling2D (bilinear) + Conv2D ×2 |
| Synthesis output | Mean only, or DLMM (4-mixture logistic) | Mean + scale (sigma) from shared trunk |
| Factorized density | 3-layer CDF network per channel | Same (FactorizedPrior — identical algorithm) |
| Sigma floor | MIN_SCALE (library-defined) | MIN_SCALE = 0.11 |

**Why:** 128-channel hyperprior captures the latent distribution adequately at 96-channel latents. The DLMM (discrete logistic mixture model) of the original adds quality at the cost of complexity; a simple Gaussian conditional is sufficient and produces smaller TFLite models.

---

## Entropy Coding

| | Original | Rebuild |
|---|---|---|
| Library | Custom rANS (src/compression/ans.py) | Same code — direct copy, zero changes |
| Bitstream format | .hfc binary | Same .hfc format |
| Prior for hyperlatents | Factorized non-parametric CDF | Same |
| Prior for latents | Gaussian conditional (scale table) | Same |
| Runtime dependency | PyTorch tensors as inputs | NumPy arrays as inputs |
| GPU acceleration | CUDA via PyTorch | None (CPU numpy on Pi) |

The entropy coding layer is fully reused. The only change is stripping `torch.Tensor` calls and replacing them with `np.array` equivalents in the adapter modules.

---

## Loss Functions

| | Original | Rebuild |
|---|---|---|
| Rate | Adaptive λ_A / λ_B on bpp | Same adaptive scheme |
| Distortion | MSE (k_M = 0.075 × 2⁻⁵) | Same (k_M identical) |
| Perceptual | LPIPS (AlexNet backbone) | MS-SSIM (`tf.image.ssim_multiscale`) |
| GAN (Phase 2) | Non-saturating + least-squares variants, β=0.15 | Non-saturating, β=0.15 |
| Discriminator input | Image + upscaled latents concatenated (15 ch) | Image only (3 ch) |
| Discriminator arch | PatchGAN with spectral norm, 64→128→256→512→1 | PatchGAN with L2 instance norm, 64→128→256→1 |

**Why LPIPS → MS-SSIM:** LPIPS depends on a pretrained AlexNet that is not trivially deployable in TF and adds ~60 MB of non-training weights. MS-SSIM is built into TensorFlow, produces comparable perceptual-quality signal at low bitrates, and requires no additional dependencies on the Pi.

---

## Training

| | Original | Rebuild |
|---|---|---|
| Phase 1 steps | ~1M (compression + LPIPS) | 500k (compression + MS-SSIM) |
| Phase 2 steps | ~1M (GAN fine-tuning) | 200k (GAN fine-tuning) |
| Batch size | 8 | 8 |
| Base LR | 1e-4 | 1e-4 |
| LR decay | ×0.1 at step 500k | ×0.1 at step 500k |
| Backbone LR | N/A (no pretrained backbone) | 1e-5 (10× slower than projection head) |
| Optimizers | Adam × 3 (amort, entropy, disc) | Adam × 3 (same split) |
| Gradient clipping | None | Global norm 5.0 |
| Quantization noise | U[−0.5, 0.5] additive (training) | Same |
| Hard quantize | torch.round + STE | tf.custom_gradient round + STE |
| Checkpointing | Custom torch save | tf.train.CheckpointManager |
| Lambda schedule | λ_A=2.0→1.0 at step 50k (warm-up) | λ_A=2.0 fixed per regime |

---

## Deployment Pipeline

| Stage | Original | Rebuild |
|-------|---------|---------|
| Export | None — runs PyTorch directly | SavedModel → TFLite flatbuffer |
| Quantization | None | INT8 post-training (200-image calibration set) |
| Runtime | Full PyTorch install (~2 GB) | `tflite-runtime` (~5 MB pip package) |
| Inference entry point | `compress.py` (GPU) | `tflite/inference/compress.py` (Pi CPU) |
| Thread count | Single (GPU serial) | 4 threads (Pi4 Cortex-A72 cores) |

---

## Parameter Count (approximate)

| Component | Original | Rebuild |
|-----------|---------|---------|
| Encoder | ~12M | ~1.5M (MobileNetV3) + ~9K (projection) |
| Decoder | ~25M | ~1.2M |
| Hyperprior | ~3M | ~500K |
| Discriminator (training only) | ~5M | ~2M |
| **Total (inference)** | **~40M** | **~3.2M** |

---

## What Is Unchanged

- The rANS entropy coding algorithm (`src/compression/ans.py` copied verbatim)
- The .hfc bitstream binary format (compress/decompress interoperable)
- The rate-distortion training objective structure (λ_A / λ_B adaptive weighting)
- The two-phase training strategy (compression-only → GAN fine-tuning)
- The alternating generator/discriminator update schedule
- The factorized prior CDF network algorithm (re-implemented in TF)
- The Gaussian conditional prior with scale table indexing
- The 256×256 spatial resolution and 16× downsampling factor
- The training dataset path (`data/openimages/`)
- The three bitrate regimes: low (0.14 bpp), med (0.30 bpp), high (0.45 bpp)
