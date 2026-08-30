"""
One-time conversion: torchvision VGG16 + vendored LPIPS linear weights -> .npz

Run this ONCE, in an environment with PyTorch/torchvision. The resulting .npz
is all tflite/lpips.py needs, so evaluation never imports torch.

Why torchvision specifically, and not tf.keras.applications.VGG16: LPIPS feeds
the backbone through its ScalingLayer, and with input 2x-1 for x in [0,1] that
computes (2x - 0.970) / 0.458 == (x - 0.485) / 0.229 — exactly torchvision's
ImageNet normalisation. Keras's VGG16 was trained with Caffe-style BGR mean
subtraction and expects a different input distribution entirely, so its
features would not match the ones the LPIPS linear weights were calibrated on.

Usage:
    /home/lema/miniconda3/envs/python3.8/bin/python \
        -m tflite.convert_lpips_weights \
        --out tflite/lpips_weights/vgg16_lpips.npz
"""

import argparse
import os

import numpy as np
import torch
import torchvision

# torchvision VGG16 `features` indices whose output LPIPS taps (post-ReLU):
# relu1_2, relu2_2, relu3_3, relu4_3, relu5_3
TAPS = (3, 8, 15, 22, 29)
N_CONV = 13          # conv layers up to and including conv5_3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="tflite/lpips_weights/vgg16_lpips.npz")
    p.add_argument("--lin_weights",
                   default="src/loss/perceptual_similarity/weights/v0.1/vgg.pth")
    args = p.parse_args()

    print("Loading torchvision VGG16 (downloads on first run) ...")
    feats = torchvision.models.vgg16(pretrained=True).features

    out = {}

    # VGG16 `features` is a fixed stack: blocks of [2,2,3,3,3] conv+ReLU with a
    # 2x2 maxpool after each block. LPIPS taps the last ReLU of every block, so
    # tflite/lpips.py rebuilds it from BLOCKS alone and needs only the conv
    # weights here.
    conv_i = 0
    for layer in feats:
        if isinstance(layer, torch.nn.Conv2d):
            out[f"conv{conv_i}_w"] = (
                layer.weight.detach().numpy().transpose(2, 3, 1, 0).astype(np.float32)
            )
            out[f"conv{conv_i}_b"] = layer.bias.detach().numpy().astype(np.float32)
            conv_i += 1
            if conv_i == N_CONV:
                break
    assert conv_i == N_CONV, f"expected {N_CONV} convs, found {conv_i}"

    # --- LPIPS linear weights: (1,C,1,1) -> (C,) ---
    print(f"Loading linear weights from {args.lin_weights}")
    sd = torch.load(args.lin_weights, map_location="cpu")
    for i in range(len(TAPS)):
        key = f"lin{i}.model.1.weight"
        out[f"lin{i}"] = sd[key].detach().numpy().reshape(-1).astype(np.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **out)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\nWrote {args.out}  ({size_mb:.1f} MB)")
    for i in range(len(TAPS)):
        print(f"  lin{i}: {out[f'lin{i}'].shape}")


if __name__ == "__main__":
    main()
