import numpy as np
import pandas as pd
import os, glob, time
import logging, argparse
import functools

from pprint import pprint
from tqdm import tqdm, trange
from collections import defaultdict, namedtuple

import torch
import torch.quantization
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# Custom modules
from src.helpers import utils, datasets, metrics
from src.compression import compression_utils
from src.loss.perceptual_similarity import perceptual_loss as ps
from default_config import hific_args, mse_lpips_args, directories, ModelModes, ModelTypes
from default_config import args as default_args

_QENGINE_PRIORITY = ['x86', 'qnnpack', 'fbgemm', 'onednn']
_available_engines = torch.backends.quantized.supported_engines
_QENGINE = next((e for e in _QENGINE_PRIORITY if e in _available_engines), 'none')
torch.backends.quantized.engine = _QENGINE

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
    model.Hyperprior.synthesis_mu.half()
    model.Hyperprior.synthesis_std.half()
    # hyperprior_entropy_model and prior_entropy_model CDF tables are int32 — untouched by .half()
    # hyperlatent_likelihood stays FP32; cdf_logits casts its input to float internally

def _convert_model_to_int8(model):
    # Dynamic PTQ: quantizes Conv2d/ConvTranspose2d weights to int8.
    # Custom layers (ChannelNorm2D, residual adds, LowerBoundToward) are transparent to this.
    # Quantized modules require CPU — callers must move the model to CPU after this call.
    quantize = lambda m: torch.quantization.quantize_dynamic(
        m, {nn.Conv2d, nn.ConvTranspose2d, nn.Linear}, dtype=torch.qint8
    )
    model.Encoder = quantize(model.Encoder)
    if hasattr(model, 'Generator'):  # absent when use_stripped_model=True
        model.Generator = quantize(model.Generator)
    model.Hyperprior.analysis_net = quantize(model.Hyperprior.analysis_net)
    model.Hyperprior.synthesis_mu = quantize(model.Hyperprior.synthesis_mu)
    model.Hyperprior.synthesis_std = quantize(model.Hyperprior.synthesis_std)
    # Entropy coding infrastructure (hyperlatent_likelihood, entropy models) stays FP32

def prepare_model(ckpt_path, use_fp16=False, use_int8=False, quantize=False):

    make_deterministic()
    device = utils.get_device()
    # logger = utils.logger_setup(logpath=os.path.join(input_dir, f'logs_{time.time()}'), filepath=os.path.abspath(__file__))
    logger = utils.logger_setup(logpath=os.path.join("data/test_inputs", f'logs_{time.time()}'), filepath=os.path.abspath(__file__))
    loaded_args, model, _ = utils.load_model(ckpt_path, logger, device, model_mode=ModelModes.EVALUATION,
        current_args_d=None, prediction=True, strict=False, silent=True)
    model.logger.info('Model loaded from disk.')

    # Build probability tables first, while model is on CUDA.
    # estimate_tails() inside build_tables() always creates tensors on CUDA via utils.get_device(),
    # so the density model parameters must still be on CUDA at this point.
    model.logger.info('Building hyperprior probability tables...')
    if hasattr(model.Hyperprior, 'hyperprior_entropy_model'):
        model.Hyperprior.hyperprior_entropy_model.build_tables()
    model.logger.info('All tables built.')

    if use_fp16 and torch.cuda.is_available():
        _convert_model_to_fp16(model)
        model.logger.info('Model converted to FP16.')
    elif use_int8:
        model.cpu()  # quantize_dynamic requires CPU; safe to move after build_tables()
        _convert_model_to_int8(model)
        model.logger.info('Model converted to dynamic Int8 (CPU).')
    elif quantize:
        model.cpu()
        model.eval()
        # The QAT observers/fake quants are already injected in load_model
        torch.ao.quantization.convert(model, inplace=True)
        model.logger.info('Model converted to static QAT Int8.')

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

    # We load the unquantized/qat model and convert it properly below if --quantize is provided
    loaded_args, model, _ = utils.load_model(args.ckpt_path, logger, device, model_mode=ModelModes.EVALUATION,
        current_args_d=None, prediction=True, strict=False)

    if args.quantize:
        model.eval()
        # The QAT observers/fake quants are already injected in load_model
        # Convert QAT to Static Quantized model
        torch.ao.quantization.convert(model, inplace=True)

    # Override current arguments with recorded
    dictify = lambda x: dict((n, getattr(x, n)) for n in dir(x) if not (n.startswith('__') or 'logger' in n))
    loaded_args_d, args_d = dictify(loaded_args), dictify(args)
    loaded_args_d.update(args_d)
    args = utils.Struct(**loaded_args_d)
    logger.info(loaded_args_d)

    # Build probability tables first, before any dtype/device conversion (same reason as prepare_model)
    logger.info('Building hyperprior probability tables...')
    if hasattr(model.Hyperprior, 'hyperprior_entropy_model'):
        model.Hyperprior.hyperprior_entropy_model.build_tables()
    logger.info('All tables built.')

    if getattr(args, 'fp16', False) and torch.cuda.is_available():
        _convert_model_to_fp16(model)
        logger.info('Model converted to FP16.')
    elif getattr(args, 'int8', False):
        model.cpu()  # quantize_dynamic requires CPU; safe to move after build_tables()
        device = torch.device('cpu')
        _convert_model_to_int8(model)
        logger.info('Model converted to dynamic Int8 (CPU).')
    elif getattr(args, 'quantize', False):
        model.cpu()
        device = torch.device('cpu')
        model.eval()
        torch.ao.quantization.convert(model, inplace=True)
        logger.info('Model converted to static QAT Int8.')


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
                              use_int8=getattr(args, 'int8', False),
                              quantize=getattr(args, 'quantize', False))
    filename = os.path.basename(args.compressed_file)
    out_path=os.path.join(args.output_dir, f"{filename}.png")
    load_and_decompress(model, args.compressed_file, out_path)

def compress(args):
    model, loaded_args = prepare_model(args.ckpt_path,
                                       use_fp16=getattr(args, 'fp16', False),
                                       use_int8=getattr(args, 'int8', False),
                                       quantize=getattr(args, 'quantize', False))

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
        help="Quantize Conv weights to Int8 for CPU inference (entropy coding stays FP32)")
    parser.add_argument("--quantize", action="store_true", help="Enable static quantization for inference.")
    args = parser.parse_args()

    assert not (args.fp16 and args.int8), "--fp16 and --int8 are mutually exclusive"
    assert not (args.fp16 and args.quantize), "--fp16 and --quantize are mutually exclusive"

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
