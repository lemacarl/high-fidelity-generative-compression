import numpy as np
import pandas as pd
import os, glob, time
import logging, argparse
import functools

from pprint import pprint
from tqdm import tqdm, trange
from collections import defaultdict, namedtuple

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# Custom modules
from src.helpers import utils, datasets, metrics
from src.compression import compression_utils
from src.loss.perceptual_similarity import perceptual_loss as ps
from default_config import hific_args, mse_lpips_args, directories, ModelModes, ModelTypes
from default_config import args as default_args

File = namedtuple('File', ['original_path', 'compressed_path',
                           'compressed_num_bytes', 'bpp'])

def make_deterministic(seed=42):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Don't go fast boi :(

    np.random.seed(seed)

def prepare_dataloader(args, input_dir, output_dir, batch_size=1):

    # `batch_size` must be 1 for images of different shapes
    input_images = glob.glob(os.path.join(input_dir, '*.jpg'))
    input_images += glob.glob(os.path.join(input_dir, '*.png'))
    # assert len(input_images) > 0, 'No valid image files found in supplied directory!'
    print('Input images')
    pprint(input_images)

    eval_loader = datasets.get_dataloaders('evaluation', root=input_dir, batch_size=batch_size,
                                           logger=None, shuffle=False, normalize=args.normalize_input_image, single=args.single)
    utils.makedirs(output_dir)

    return eval_loader

def _convert_model_to_fp16(model):
    model.Encoder.half()
    if hasattr(model, 'Generator'):  # absent when use_stripped_model=True
        model.Generator.half()
    model.Hyperprior.analysis_net.half()
    if hasattr(model.Hyperprior, 'synthesis_mu'):
        model.Hyperprior.synthesis_mu.half()
        model.Hyperprior.synthesis_std.half()
    elif hasattr(model.Hyperprior, 'synthesis_DLMM_params'):
        model.Hyperprior.synthesis_DLMM_params.half()
    # hyperprior_entropy_model and prior_entropy_model CDF tables are int32 — untouched by .half()
    # hyperlatent_likelihood stays FP32; cdf_logits casts its input to float internally

def _convert_qat_model_to_int8(model, checkpoint, loaded_args):
    """
    Finalises a QAT-trained model to true int8 via convert_fx().

    load_model() creates a plain FP32 model, so we must:
      1. Re-prepare each submodule with prepare_qat_fx (creates GraphModules)
      2. Reload the QAT state dict so fake-quantize scale/zero_point are restored
      3. Call convert_fx() to lower to true int8

    Requirements:
    - model must be on CPU (int8 kernels are CPU-only).
    - checkpoint must have been saved with qat_active=True.
    """
    from src.quantization.qat_utils import prepare_net_for_qat, build_qconfig_mapping, convert_net_to_int8

    image_dims = getattr(loaded_args, 'image_dims', (3, 256, 256))
    # Use the backend from the checkpoint, but fall back to qnnpack on ARM (e.g. Raspberry Pi)
    # where x86/fbgemm is unavailable.
    backend = getattr(loaded_args, 'qat_backend', 'x86')
    supported = torch.backends.quantized.supported_engines
    if backend not in supported:
        backend = 'qnnpack' if 'qnnpack' in supported else supported[0]
    torch.backends.quantized.engine = backend
    qcm = build_qconfig_mapping(backend)

    # Save plain Python attributes stripped by FX tracing (both prepare and convert).
    enc_n_down = model.Encoder.n_downsampling_layers
    ana_n_down = model.Hyperprior.analysis_net.n_downsampling_layers

    # --- Step 1: re-prepare plain modules as GraphModules ---
    dummy_img = torch.zeros(1, *image_dims)
    model.train()  # prepare_qat_fx requires train mode

    model.Encoder = prepare_net_for_qat(model.Encoder, dummy_img, qcm, backend)
    with torch.no_grad():
        lat = model.Encoder(dummy_img).detach()

    if hasattr(model, 'Generator'):
        model.Generator = prepare_net_for_qat(model.Generator, lat, qcm, backend)

    model.Hyperprior.analysis_net = prepare_net_for_qat(
        model.Hyperprior.analysis_net, lat, qcm, backend)
    with torch.no_grad():
        hyp = model.Hyperprior.analysis_net(lat).detach()

    if hasattr(model.Hyperprior, 'synthesis_mu'):
        model.Hyperprior.synthesis_mu = prepare_net_for_qat(
            model.Hyperprior.synthesis_mu, hyp, qcm, backend)
        model.Hyperprior.synthesis_std = prepare_net_for_qat(
            model.Hyperprior.synthesis_std, hyp, qcm, backend)
    elif hasattr(model.Hyperprior, 'synthesis_DLMM_params'):
        model.Hyperprior.synthesis_DLMM_params = prepare_net_for_qat(
            model.Hyperprior.synthesis_DLMM_params, hyp, qcm, backend)

    # --- Step 2: reload QAT weights (restores fake-quantize scale/zero_point) ---
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # --- Step 3: convert GraphModules to true int8 ---
    model.eval()
    model.Encoder = convert_net_to_int8(model.Encoder)
    model.Encoder.n_downsampling_layers = enc_n_down

    if hasattr(model, 'Generator'):
        model.Generator = convert_net_to_int8(model.Generator)

    model.Hyperprior.analysis_net = convert_net_to_int8(model.Hyperprior.analysis_net)
    model.Hyperprior.analysis_net.n_downsampling_layers = ana_n_down

    if hasattr(model.Hyperprior, 'synthesis_mu'):
        model.Hyperprior.synthesis_mu = convert_net_to_int8(model.Hyperprior.synthesis_mu)
        model.Hyperprior.synthesis_std = convert_net_to_int8(model.Hyperprior.synthesis_std)
    elif hasattr(model.Hyperprior, 'synthesis_DLMM_params'):
        model.Hyperprior.synthesis_DLMM_params = convert_net_to_int8(model.Hyperprior.synthesis_DLMM_params)
    # Entropy coding infrastructure (hyperlatent_likelihood, entropy models) stays FP32/int32


def prepare_model(ckpt_path, use_fp16=False, use_int8=False):

    make_deterministic()
    device = utils.get_device()
    logger = utils.logger_setup(logpath=os.path.join("data/test_inputs", f'logs_{time.time()}'), filepath=os.path.abspath(__file__))
    loaded_args, model, _ = utils.load_model(ckpt_path, logger, device, model_mode=ModelModes.EVALUATION,
        current_args_d=None, prediction=True, strict=False, silent=True)
    model.logger.info('Model loaded from disk.')

    # Build probability tables while model is on CUDA (estimate_tails uses get_device()).
    model.logger.info('Building hyperprior probability tables...')
    model.Hyperprior.hyperprior_entropy_model.build_tables()
    model.logger.info('All tables built.')

    if use_fp16 and torch.cuda.is_available():
        _convert_model_to_fp16(model)
        model.logger.info('Model converted to FP16.')
    elif use_int8:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        assert checkpoint.get('qat_active', False), (
            '--int8 requires a checkpoint trained with --qat. '
            'Re-train using the --qat flag to enable QAT.')
        model.cpu()
        _convert_qat_model_to_int8(model, checkpoint, loaded_args)
        model.logger.info('Model converted to QAT int8 (CPU).')

    return model, loaded_args

def compress_and_save(model, args, data_loader, output_dir):
    # Compress and save compressed format to disk

    use_fp16 = getattr(args, 'fp16', False)
    use_int8 = getattr(args, 'int8', False)
    if use_int8:
        device = torch.device('cpu')
    else:
        device = utils.get_device()
    input_dtype = torch.float16 if (use_fp16 and torch.cuda.is_available()) else torch.float

    model.logger.info('Starting compression...')

    with torch.no_grad():
        for idx, (data, bpp, filenames) in enumerate(tqdm(data_loader), 0):
            data = data.to(device, dtype=input_dtype)
            assert data.size(0) == 1, 'Currently only supports saving single images.'

            # Perform entropy coding
            compressed_output = model.compress(data)

            out_path = os.path.join(output_dir, f"{filenames[0]}_{args.name}_compressed.hfc")
            actual_bpp, theoretical_bpp = compression_utils.save_compressed_format(compressed_output,
                out_path=out_path)
            model.logger.info(f'Attained: {actual_bpp:.3f} bpp vs. theoretical: {theoretical_bpp:.3f} bpp.')


def load_and_decompress(model, compressed_format_path, out_path):
    # Decompress single image from compressed format on disk

    compressed_output = compression_utils.load_compressed_format(compressed_format_path)
    start_time = time.time()
    with torch.no_grad():
        reconstruction = model.decompress(compressed_output)

    torchvision.utils.save_image(reconstruction.float(), out_path, normalize=True)
    delta_t = time.time() - start_time
    model.logger.info('Decoding time: {:.2f} s'.format(delta_t))
    model.logger.info(f'Reconstruction saved to {out_path}')

    return reconstruction

def compress_and_decompress(args):

    # Reproducibility
    make_deterministic()
    perceptual_loss_fn = ps.PerceptualLoss(model='net-lin', net='alex', use_gpu=torch.cuda.is_available())

    # Load model
    device = utils.get_device()
    logger = utils.logger_setup(logpath=os.path.join(args.image_dir, 'logs'), filepath=os.path.abspath(__file__))
    loaded_args, model, _ = utils.load_model(args.ckpt_path, logger, device, model_mode=ModelModes.EVALUATION,
        current_args_d=None, prediction=True, strict=False)

    # Override current arguments with recorded
    dictify = lambda x: dict((n, getattr(x, n)) for n in dir(x) if not (n.startswith('__') or 'logger' in n))
    loaded_args_d, args_d = dictify(loaded_args), dictify(args)
    loaded_args_d.update(args_d)
    args = utils.Struct(**loaded_args_d)
    logger.info(loaded_args_d)

    # Build probability tables first, before any dtype/device conversion (same reason as prepare_model)
    logger.info('Building hyperprior probability tables...')
    model.Hyperprior.hyperprior_entropy_model.build_tables()
    logger.info('All tables built.')

    if getattr(args, 'fp16', False) and torch.cuda.is_available():
        _convert_model_to_fp16(model)
        logger.info('Model converted to FP16.')
    elif getattr(args, 'int8', False):
        checkpoint = torch.load(args.ckpt_path, map_location='cpu')
        assert checkpoint.get('qat_active', False), (
            '--int8 requires a checkpoint trained with --qat. '
            'Re-train using the --qat flag to enable QAT.')
        model.cpu()
        device = torch.device('cpu')
        _convert_qat_model_to_int8(model, checkpoint, loaded_args)
        logger.info('Model converted to QAT int8 (CPU).')


    eval_loader = datasets.get_dataloaders('evaluation', root=args.image_dir, batch_size=args.batch_size,
                                           logger=logger, shuffle=False, normalize=args.normalize_input_image)

    n, N = 0, len(eval_loader.dataset)
    input_filenames_total = list()
    output_filenames_total = list()
    bpp_total, q_bpp_total, LPIPS_total = torch.Tensor(N), torch.Tensor(N), torch.Tensor(N)
    MS_SSIM_total, PSNR_total = torch.Tensor(N), torch.Tensor(N)
    max_value = 255.
    MS_SSIM_func = metrics.MS_SSIM(data_range=max_value)
    utils.makedirs(args.output_dir)

    logger.info('Starting compression...')
    start_time = time.time()

    use_fp16 = getattr(args, 'fp16', False)
    input_dtype = torch.float16 if (use_fp16 and torch.cuda.is_available()) else torch.float

    with torch.no_grad():

        for idx, (data, bpp, filenames) in enumerate(tqdm(eval_loader), 0):
            data = data.to(device, dtype=input_dtype)
            B = data.size(0)
            input_filenames_total.extend(filenames)

            if args.reconstruct is True:
                # Reconstruction without compression
                reconstruction, q_bpp = model(data, writeout=False)
            else:
                # Perform entropy coding
                compressed_output = model.compress(data)

                if args.save is True:
                    assert B == 1, 'Currently only supports saving single images.'
                    compression_utils.save_compressed_format(compressed_output,
                        out_path=os.path.join(args.output_dir, f"{filenames[0]}_compressed.hfc"))

                reconstruction = model.decompress(compressed_output)
                q_bpp = compressed_output.total_bpp

            reconstruction = reconstruction.float()

            if args.normalize_input_image is True:
                # [-1., 1.] -> [0., 1.]
                data = (data + 1.) / 2.

            perceptual_loss = perceptual_loss_fn.forward(reconstruction, data.float(), normalize=True)

            if args.metrics is True:
                # [0., 1.] -> [0., 255.]
                psnr = metrics.psnr(reconstruction.cpu().numpy() * max_value, data.float().cpu().numpy() * max_value, max_value)
                ms_ssim = MS_SSIM_func(reconstruction * max_value, data.float() * max_value)
                PSNR_total[n:n + B] = torch.Tensor(psnr)
                MS_SSIM_total[n:n + B] = ms_ssim.data

            for subidx in range(reconstruction.shape[0]):
                if B > 1:
                    q_bpp_per_im = float(q_bpp.cpu().numpy()[subidx])
                else:
                    q_bpp_per_im = float(q_bpp.item()) if type(q_bpp) == torch.Tensor else float(q_bpp)

                fname = os.path.join(args.output_dir, "{}_RECON_{:.3f}bpp.png".format(filenames[subidx], q_bpp_per_im))
                torchvision.utils.save_image(reconstruction[subidx], fname, normalize=True)
                output_filenames_total.append(fname)

            bpp_total[n:n + B] = bpp.data
            q_bpp_total[n:n + B] = q_bpp.data if type(q_bpp) == torch.Tensor else q_bpp
            LPIPS_total[n:n + B] = perceptual_loss.data
            n += B

    df = pd.DataFrame([input_filenames_total, output_filenames_total]).T
    df.columns = ['input_filename', 'output_filename']
    df['bpp_original'] = bpp_total.cpu().numpy()
    df['q_bpp'] = q_bpp_total.cpu().numpy()
    df['LPIPS'] = LPIPS_total.cpu().numpy()

    if args.metrics is True:
        df['PSNR'] = PSNR_total.cpu().numpy()
        df['MS_SSIM'] = MS_SSIM_total.cpu().numpy()

    df_path = os.path.join(args.output_dir, 'compression_metrics.h5')
    df.to_hdf(df_path, key='df')

    pprint(df)

    logger.info('Complete. Reconstructions saved to {}. Output statistics saved to {}'.format(args.output_dir, df_path))
    delta_t = time.time() - start_time
    logger.info('Time elapsed: {:.3f} s'.format(delta_t))
    logger.info('Rate: {:.3f} Images / s:'.format(float(N) / delta_t))


def decompress(args):
    assert args.compressed_file and os.path.isfile(args.compressed_file), "The compressed file is not set or does not exist."
    model, _ = prepare_model(args.ckpt_path,
                              use_fp16=getattr(args, 'fp16', False),
                              use_int8=getattr(args, 'int8', False))
    filename = os.path.basename(args.compressed_file)
    out_path=os.path.join(args.output_dir, f"{filename}.png")
    load_and_decompress(model, args.compressed_file, out_path)

def compress(args):
    model, loaded_args = prepare_model(args.ckpt_path,
                                       use_fp16=getattr(args, 'fp16', False),
                                       use_int8=getattr(args, 'int8', False))

    # Override current arguments with recorded
    dictify = lambda x: dict((n, getattr(x, n)) for n in dir(x) if not (n.startswith('__') or 'logger' in n))
    loaded_args_d, args_d = dictify(loaded_args), dictify(args)
    loaded_args_d.update(args_d)
    args = utils.Struct(**loaded_args_d)

    dataloader = prepare_dataloader(args, args.image_dir, args.output_dir)
    compress_and_save(model, args, dataloader, args.output_dir)

def main(**kwargs):

    description = "Compresses batch of images using learned model specified via -ckpt argument."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ckpt", "--ckpt_path", type=str, required=True, help="Path to model to be restored")
    parser.add_argument("-i", "--image_dir", type=str, default='data/originals',
        help="Path to directory containing images to compress")
    parser.add_argument("-o", "--output_dir", type=str, default='data/reconstructions',
        help="Path to directory to store output images")
    parser.add_argument('-bs', '--batch_size', type=int, default=1,
        help="Loader batch size. Set to 1 if images in directory are different sizes.")
    parser.add_argument("-rc", "--reconstruct", help="Reconstruct input image without compression.", action="store_true")
    parser.add_argument("-save", "--save", help="Save compressed format to disk.", action="store_true")
    parser.add_argument("-metrics", "--metrics", help="Evaluate compression metrics.", action="store_true")
    parser.add_argument("-d", "--decompress", help="Decompress the compressed file.", action="store_true")
    parser.add_argument("-c", "--compress", help="Compress input file.", action="store_true")
    parser.add_argument("-cf", "--compressed_file", type=str, help="Path to compressed file to decompress")
    parser.add_argument("-s", "--single", help="Compress single file", action='store_true', default=False)
    parser.add_argument("--fp16", action="store_true",
        help="Convert model to FP16 for faster GPU inference (entropy coding stays FP32)")
    parser.add_argument("--int8", action="store_true",
        help="Convert a QAT-trained checkpoint to true int8 for CPU inference. "
             "Requires the checkpoint to have been trained with --qat. "
             "Entropy coding stays FP32/int32.")
    args = parser.parse_args()

    assert not (args.fp16 and args.int8), "--fp16 and --int8 are mutually exclusive"

    input_images = glob.glob(os.path.join(args.image_dir, '*.jpg'))
    input_images += glob.glob(os.path.join(args.image_dir, '*.png'))


    print('Input images')
    pprint(input_images)

    if args.decompress is True:
        decompress(args)
    elif args.compress is True:
        compress(args)
    else:
        assert len(input_images) > 0, 'No valid image files found in supplied directory!'
        compress_and_decompress(args)

if __name__ == '__main__':
    main()
