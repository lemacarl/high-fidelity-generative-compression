# TFLite Generative Compression

A complete TensorFlow/Keras reimplementation of [High-Fidelity Generative Image Compression](https://arxiv.org/abs/2006.09965) targeting edge deployment on Raspberry Pi. Produces four standalone `.tflite` models that run inference without a GPU.

## Architecture

The pipeline mirrors the original PyTorch implementation but with substitutions for TFLite/INT8/ARM compatibility:

| Component | Original (`src/`) | This path (`tflite/`) |
|-----------|------------------|----------------------|
| Encoder | Custom 4-layer conv | MobileNetV3Small + 1×1 projection |
| Latent channels | 220 | 96 |
| Hyper channels | 128 | 192 |
| Upsampling | ConvTranspose2D | UpSampling2D (bilinear) + Conv2D |
| Normalization | ChannelNorm / InstanceNorm | LayerNormalization |
| Padding | ReflectionPad2d | `"same"` |

**Why these substitutions?** `ConvTranspose2D` produces checkerboard artefacts after INT8 quantization on ARM. `LayerNormalization` has no running-stats state, which avoids the NaN BatchNorm export failure (see [Export](#export-to-tflite)). The decoder matches the original HIFIC Generator at ~25M parameters.

The discriminator (used only in Phase 2 GAN training) is a full-capacity PatchGAN with SpectralNormalization matching `src/network/discriminator.py`. It is never exported to TFLite.

### Diagrams

Layer-level diagrams of the three networks that are not a stock backbone:

- [Decoder](../assets/decoder_architecture.svg) — latents (B,16,16,96) through 9 ResBlocks and 4 bilinear upsamples to RGB
- [Hyperprior](../assets/hyperprior_architecture.svg) — hyperencoder, factorized prior over z, and the hyperdecoder producing per-latent (mu, sigma)
- [Discriminator](../assets/discriminator_architecture.svg) — conditional PatchGAN taking (image, latents)

The encoder is MobileNetV3Small as shipped by Keras plus a 1x1 projection, so it has no diagram here.

### Module map

```
tflite/
  model/
    encoder.py          MobileNetV3Small backbone + projection head → latents (B,16,16,96)
    decoder.py          9× ResBlock + 4× bilinear upsample → RGB (B,256,256,3)
    hyperprior.py       Hyper encoder/decoder + FactorizedPrior layer
    compression_model.py  CompressionModel: ties encoder/decoder/hyperprior together
  training/
    trainer.py          Training loop (Phase 1 + Phase 2 GAN), CLI entry point
    losses.py           Rate (adaptive λ), distortion (MSE), perceptual (MS-SSIM), GAN losses
    data_pipeline.py    tf.data pipeline; random crop+flip for training, centre-crop for eval
  conversion/
    export_tflite.py    Export Keras → FP32 / INT8 .tflite; BN sanitization + verification
  inference/
    compress.py         On-device compress/decompress CLI (works with tflite-runtime on Pi)
  compression/
    entropy_models.py   FactorizedPriorNumpy — rebuilds ANS CDF tables from density_weights.npz
    prior_model.py      Gaussian prior for latent ANS coding
    entropy_coding.py   Pure-NumPy vectorized ANS encoder/decoder (no PyTorch)
    compression_utils.py  .hfc binary format I/O
    ans.py              Low-level rANS primitives
  evaluate.py           GPU-side evaluation: PSNR, MS-SSIM, BPP (no TFLite conversion needed)
```

## Dependencies

```bash
# GPU workstation (training + export)
pip install -r tflite/requirements_tflite.txt

# Raspberry Pi (inference only)
pip install tflite-runtime Pillow numpy scipy
```

## Training

Two-phase training. Phase 1 converges the rate-distortion model; Phase 2 fine-tunes with GAN adversarial loss for perceptual quality.

**Phase 1 — compression only**

```bash
python -m tflite.training.trainer \
    --dataset_path data/openimages \
    --regime low \
    --n_steps 500000 \
    --checkpoint_dir experiments/tflite_low/
```

**Phase 2 — GAN fine-tuning** (warm-start from Phase 1)

```bash
python -m tflite.training.trainer \
    --dataset_path data/openimages \
    --regime low \
    --model_type compression_gan \
    --warmstart \
    --checkpoint experiments/tflite_low/ckpt-500000 \
    --n_steps 200000 \
    --checkpoint_dir experiments/tflite_low_gan/
```

**Regime targets:**

| Regime | Target BPP | λ_a |
|--------|-----------|-----|
| `low`  | 0.14      | 2.0 |
| `med`  | 0.30      | 1.0 |
| `high` | 0.45      | 0.5 |

Training checkpoints land in `--checkpoint_dir`. After the final step, `density_weights.npz` is written to the same directory — this file is required for ANS entropy coding at inference time and must be copied alongside the `.tflite` models.

**TensorBoard:**

```bash
tensorboard --logdir experiments/tflite_low/logs
```

## Evaluation (GPU)

Run compress→decompress entirely in TF without converting to TFLite first:

```bash
python -m tflite.evaluate \
    --checkpoint experiments/tflite_low/final-500000 \
    --images data/test_inputs/image.png \
    --out_dir eval_out/
```

Outputs PSNR, MS-SSIM, BPP, and a side-by-side PNG of original vs. reconstruction.

## Export to TFLite

```bash
# FP32
python -m tflite.conversion.export_tflite \
    --checkpoint experiments/tflite_low/final-500000 \
    --out_dir tflite_models/

# FP32 + INT8 (requires ~100 calibration images)
python -m tflite.conversion.export_tflite \
    --checkpoint experiments/tflite_low/final-500000 \
    --out_dir tflite_models/ \
    --int8 \
    --image_dir data/train/
```

Produces `encoder.tflite`, `hyper_encoder.tflite`, `hyper_decoder.tflite`, `decoder.tflite` (and `*_int8.tflite` variants). The exporter automatically audits for NaN/Inf in weights and sanitizes stale BatchNorm moving stats — a known failure mode where `training=False` exposes NaN stats that were masked during training.

Copy both the `.tflite` files and `density_weights.npz` to the target device.

## Inference on device (Raspberry Pi)

```bash
# Compress
python -m tflite.inference.compress --compress \
    --input photo.jpg \
    --output photo.hfc \
    --models_dir ~/compression/tflite_models/ \
    --density_weights ~/compression/density_weights.npz

# Decompress
python -m tflite.inference.compress --decompress \
    --input photo.hfc \
    --output photo_recon.png \
    --models_dir ~/compression/tflite_models/ \
    --density_weights ~/compression/density_weights.npz
```

Compress loads only `encoder + hyper_encoder + hyper_decoder`; decompress loads only `hyper_decoder + decoder`. The unused models are never loaded, which matters on memory-constrained hardware.

By default the INT8 models are used. Pass `--fp32` to force FP32.

### .hfc format

The compressed bitstream (`.hfc`) stores:
1. ANS-coded hyperlatents z (using the factorized prior from `density_weights.npz`)
2. ANS-coded latents y (using a per-pixel Gaussian conditioned on z via the hyper decoder)
3. Spatial metadata (original image shape, coding shapes)

This is the same format used by the PyTorch path in `src/compression/`.

[rANS state flow](compression/ans_flow.svg) diagrams the 64-bit encoder/decoder loop in `tflite/compression/ans.py` — how the message stack splits between the numeric head and the spilled 32-bit tail words.
