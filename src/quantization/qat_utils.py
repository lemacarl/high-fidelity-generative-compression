"""
QAT (Quantization-Aware Training) utilities using torch.ao.quantization FX-graph mode.

Applies only to the four neural network submodules:
  - Encoder
  - Generator
  - Hyperprior.analysis_net
  - Hyperprior.synthesis_mu / synthesis_std

Entropy models (hyperprior_model.py, prior_model.py, entropy_coding.py) and the
Discriminator are intentionally excluded and remain in FP32/int32.

The latent entropy quantization in hyperprior.py (_quantize / quantize_latents_st)
is the compression algorithm's own rounding and is completely orthogonal to weight
and activation int8 QAT performed here.
"""

import torch
import torch.nn as nn
from torch.ao.quantization import get_default_qat_qconfig_mapping, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx

from src.normalisation.channel import ChannelNorm2D


def build_qconfig_mapping(backend: str = 'x86') -> QConfigMapping:
    """
    Returns a QConfigMapping for FX-graph QAT.

    ChannelNorm2D and ReflectionPad2d are explicitly set to None (FP32 passthrough):
    - ChannelNorm2D: quantizing channel-wise variance to int8 corrupts rsqrt() and
      collapses normalisation accuracy.
    - ReflectionPad2d: no learnable parameters; no benefit to quantizing.
    """
    qcm = get_default_qat_qconfig_mapping(backend)
    qcm.set_object_type(ChannelNorm2D, None)
    qcm.set_object_type(nn.ReflectionPad2d, None)
    return qcm


def prepare_net_for_qat(
    net: nn.Module,
    example_input: torch.Tensor,
    qconfig_mapping: QConfigMapping = None,
    backend: str = 'x86',
) -> 'torch.fx.GraphModule':
    """
    Applies FX-graph QAT preparation to a single network module.

    Returns a GraphModule with fake-quantize nodes inserted; it is a drop-in
    replacement for the original module with the same forward() signature.

    Notes:
    - Call while the model is on GPU; fake-quantize nodes work on CUDA.
    - Call model.train() before this; fake-quantize nodes are only active in train mode.
    - Must be called AFTER the FP32 warmup phase (see --qat_warmup_steps in train.py).
    """
    if qconfig_mapping is None:
        qconfig_mapping = build_qconfig_mapping(backend)
    net.train()
    return prepare_qat_fx(net, qconfig_mapping, example_inputs=(example_input,))


def convert_net_to_int8(net: 'torch.fx.GraphModule') -> 'torch.fx.GraphModule':
    """
    Converts a QAT-prepared GraphModule to a true int8 quantized module via convert_fx().

    Requirements before calling:
    1. model.cpu() — PyTorch int8 inference kernels require CPU.
    2. model.eval() — disables fake-quantize and uses frozen scale/zero-point.

    The QAT fake-quantize observers hold activation statistics from training;
    no separate calibration dataset is required.
    """
    net.eval()
    return convert_fx(net)
