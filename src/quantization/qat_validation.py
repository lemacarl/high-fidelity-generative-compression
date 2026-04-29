"""
QAT validation utilities: traceability tests and quality benchmarks.

Run test_fx_traceability() before any GPU training to catch FX-trace failures cheaply.
Run measure_qat_psnr_delta() after training to validate int8 quality degradation.
"""

import torch
import torch.nn as nn


def test_fx_traceability(C: int = 16, N: int = 32, n_residual_blocks: int = 2) -> None:
    """
    Verifies that each quantization-target network can be traced by
    torch.ao.quantization.quantize_fx without errors, and that a forward pass
    through the QAT-prepared module produces correct output shapes.

    Uses small C/N values for speed; no GPU required.

    Raises AssertionError on failure.
    """
    from src.network.encoder import Encoder
    from src.network.generator import Generator
    from src.network.hyper import HyperpriorAnalysis, HyperpriorSynthesis
    from src.quantization.qat_utils import build_qconfig_mapping, prepare_net_for_qat

    image_dims = (3, 256, 256)
    latent_dims = (C, 16, 16)
    batch_size = 1
    qcm = build_qconfig_mapping('x86')

    nets_and_examples = [
        (
            Encoder(image_dims, batch_size, C=C, channel_norm=True),
            torch.randn(batch_size, 3, 256, 256),
            (batch_size, C, 16, 16),
        ),
        (
            Generator(latent_dims, batch_size, C=C, n_residual_blocks=n_residual_blocks,
                      channel_norm=True),
            torch.randn(batch_size, C, 16, 16),
            (batch_size, 3, 256, 256),
        ),
        (
            HyperpriorAnalysis(C=C, N=N),
            torch.randn(batch_size, C, 16, 16),
            (batch_size, N, 4, 4),
        ),
        (
            HyperpriorSynthesis(C=C, N=N),
            torch.randn(batch_size, N, 4, 4),
            (batch_size, C, 16, 16),
        ),
    ]

    for net, example, expected_shape in nets_and_examples:
        name = net.__class__.__name__
        net.train()
        prepared = prepare_net_for_qat(net, example, qcm)
        assert prepared is not None, f'{name}: prepare_net_for_qat returned None'
        with torch.no_grad():
            out = prepared(example)
        assert tuple(out.shape) == tuple(expected_shape), (
            f'{name}: expected output shape {expected_shape}, got {tuple(out.shape)}')
        print(f'PASS: {name} — FX-trace OK, output shape {tuple(out.shape)}')


def measure_qat_psnr_delta(
    fp32_model: nn.Module,
    int8_model: nn.Module,
    test_loader,
    device: torch.device,
    n_images: int = 100,
) -> dict:
    """
    Compares reconstruction PSNR between an FP32 model and a converted int8 model
    on the same test images.

    Acceptable QAT degradation for learned image compression:
      - Mean PSNR delta < 0.3 dB
      - Max PSNR delta  < 0.5 dB  (larger indicates a QAT configuration problem)

    Returns a dict with keys: mean_delta_db, max_delta_db, n_evaluated.
    """
    import numpy as np
    from src.helpers.metrics import psnr as compute_psnr

    fp32_model.eval()
    int8_model.eval()
    deltas = []

    with torch.no_grad():
        for i, (data, _, _) in enumerate(test_loader):
            if i >= n_images:
                break
            data_fp32 = data.to(device, dtype=torch.float)
            data_cpu = data.to(torch.device('cpu'), dtype=torch.float)

            recon_fp32, _ = fp32_model(data_fp32, writeout=False)
            recon_int8, _ = int8_model(data_cpu, writeout=False)

            recon_fp32_np = recon_fp32.cpu().numpy() * 255.0
            recon_int8_np = recon_int8.numpy() * 255.0

            psnr_fp32 = compute_psnr(recon_fp32_np, data_fp32.cpu().numpy() * 255.0, 255.0)
            psnr_int8 = compute_psnr(recon_int8_np, data_cpu.numpy() * 255.0, 255.0)

            for p32, p8 in zip(psnr_fp32, psnr_int8):
                deltas.append(float(p32) - float(p8))

    mean_delta = float(np.mean(deltas)) if deltas else float('nan')
    max_delta = float(np.max(deltas)) if deltas else float('nan')
    result = dict(mean_delta_db=mean_delta, max_delta_db=max_delta, n_evaluated=len(deltas))
    print(f'QAT PSNR delta — mean: {mean_delta:.4f} dB, max: {max_delta:.4f} dB '
          f'over {len(deltas)} images')
    return result


def test_int8_compress_decompress(ckpt_path: str, image_path: str) -> None:
    """
    End-to-end smoke test: load a QAT checkpoint, convert to int8, compress an
    image, decompress, and assert basic quality constraints.

    Requires: checkpoint was saved with args.qat == True.
    """
    import torchvision.transforms as T
    from PIL import Image
    from src.quantization.qat_utils import convert_net_to_int8
    from src.helpers import utils
    from default_config import ModelModes

    logger = utils.logger_setup(logpath='data/test_inputs/qat_smoke', filepath=__file__)
    device = torch.device('cpu')

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    assert checkpoint.get('qat_active', False), (
        'Checkpoint was not saved from a QAT training run (qat_active != True).')

    _, model, _ = utils.load_model(ckpt_path, logger, device,
                                    model_mode=ModelModes.EVALUATION,
                                    prediction=True, strict=False, silent=True)
    model.Hyperprior.hyperprior_entropy_model.build_tables()

    model.cpu()
    model.eval()
    model.Encoder = convert_net_to_int8(model.Encoder)
    model.Generator = convert_net_to_int8(model.Generator)
    model.Hyperprior.analysis_net = convert_net_to_int8(model.Hyperprior.analysis_net)
    model.Hyperprior.synthesis_mu = convert_net_to_int8(model.Hyperprior.synthesis_mu)
    model.Hyperprior.synthesis_std = convert_net_to_int8(model.Hyperprior.synthesis_std)

    img = Image.open(image_path).convert('RGB')
    img_tensor = T.Compose([T.CenterCrop(256), T.ToTensor()])(img).unsqueeze(0)

    with torch.no_grad():
        compressed = model.compress(img_tensor)
        reconstruction = model.decompress(compressed)

    assert not torch.isnan(reconstruction).any(), 'Reconstruction contains NaN!'
    assert reconstruction.shape == img_tensor.shape, (
        f'Shape mismatch: {reconstruction.shape} vs {img_tensor.shape}')

    psnr_val = -10 * torch.log10(torch.mean((reconstruction - img_tensor) ** 2)).item()
    assert psnr_val > 20.0, f'PSNR too low after int8 conversion: {psnr_val:.2f} dB'
    print(f'PASS: int8 compress/decompress round-trip — PSNR {psnr_val:.2f} dB')
