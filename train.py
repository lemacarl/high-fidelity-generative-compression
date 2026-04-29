"""
Learnable generative compression model modified from [1],
implemented in Pytorch.

Example usage:
python3 train.py -h

[1] Mentzer et. al., "High-Fidelity Generative Image Compression",
    arXiv:2006.09965 (2020).
"""
import numpy as np
import os, glob, time, datetime
import logging, pickle, argparse
import functools, itertools

from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.ao.quantization
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Custom modules
from src.model import Model
from src.helpers import utils, datasets
from default_config import hific_args, mse_lpips_args, directories, ModelModes, ModelTypes

# go fast boi!!
torch.backends.cudnn.benchmark = True


def _activate_qat_on_model(model, example_batch, args, logger):
    """
    Called once at step == qat_warmup_steps to replace target submodules with
    FX-graph QAT-prepared (fake-quantize) equivalents.

    Mutates model.Encoder, model.Generator, and model.Hyperprior.*_net in-place.
    Sets model._qat_active = True and model._qat_rebuild_optimizers = True so the
    caller can rebuild the amortization optimizer with a reduced QAT learning rate.

    Do NOT call this on the top-level Model — the entropy models it contains are not
    FX-traceable. Only the four leaf networks below are prepared.
    """
    from src.quantization.qat_utils import prepare_net_for_qat, build_qconfig_mapping

    torch.backends.quantized.engine = getattr(args, 'qat_backend', 'x86')
    backend = getattr(args, 'qat_backend', 'x86')
    qcm = build_qconfig_mapping(backend)
    device = next(model.Encoder.parameters()).device

    enc_in = example_batch[:1].to(device)

    logger.info('QAT: preparing Encoder...')
    model.Encoder = prepare_net_for_qat(model.Encoder, enc_in, qcm, backend)

    with torch.no_grad():
        lat = model.Encoder(enc_in).detach()

    logger.info('QAT: preparing Generator...')
    model.Generator = prepare_net_for_qat(model.Generator, lat, qcm, backend)

    logger.info('QAT: preparing Hyperprior.analysis_net...')
    model.Hyperprior.analysis_net = prepare_net_for_qat(
        model.Hyperprior.analysis_net, lat, qcm, backend)

    with torch.no_grad():
        hyp = model.Hyperprior.analysis_net(lat).detach()

    # Hyperprior variant: Hyperprior has synthesis_mu/std; HyperpriorDLMM has synthesis_DLMM_params
    if hasattr(model.Hyperprior, 'synthesis_mu'):
        logger.info('QAT: preparing Hyperprior.synthesis_mu...')
        model.Hyperprior.synthesis_mu = prepare_net_for_qat(
            model.Hyperprior.synthesis_mu, hyp, qcm, backend)

        logger.info('QAT: preparing Hyperprior.synthesis_std...')
        model.Hyperprior.synthesis_std = prepare_net_for_qat(
            model.Hyperprior.synthesis_std, hyp, qcm, backend)
    elif hasattr(model.Hyperprior, 'synthesis_DLMM_params'):
        logger.info('QAT: preparing Hyperprior.synthesis_DLMM_params...')
        model.Hyperprior.synthesis_DLMM_params = prepare_net_for_qat(
            model.Hyperprior.synthesis_DLMM_params, hyp, qcm, backend)

    # Refresh amortization_models — references are stale after module replacement.
    model.amortization_models = [model.Encoder, model.Generator]
    model.amortization_models.extend(model.Hyperprior.amortization_models)

    model._qat_active = True
    model._qat_rebuild_optimizers = True
    logger.info('QAT fake-quantize nodes are now active.')


def create_model(args, device, logger, storage, storage_test):

    start_time = time.time()
    model = Model(args, logger, storage, storage_test, model_type=args.model_type)
    logger.info(model)
    # logger.info('Trainable parameters:')

    # for n, p in model.named_parameters():
    #     logger.info('{} - {}'.format(n, p.shape))

    logger.info("Number of trainable parameters: {}".format(utils.count_parameters(model)))
    logger.info("Estimated size (under fp32): {:.3f} MB".format(utils.count_parameters(model) * 4. / 10**6))
    logger.info('Model init {:.3f}s'.format(time.time() - start_time))

    return model

def optimize_loss(loss, opt, retain_graph=False):
    loss.backward(retain_graph=retain_graph)
    opt.step()
    opt.zero_grad()

def optimize_compression_loss(compression_loss, amortization_opt, hyperlatent_likelihood_opt):
    compression_loss.backward()
    amortization_opt.step()
    hyperlatent_likelihood_opt.step()
    amortization_opt.zero_grad()
    hyperlatent_likelihood_opt.zero_grad()

def test(args, model, epoch, idx, data, test_data, test_bpp, device, epoch_test_loss, storage, best_test_loss,
         start_time, epoch_start_time, logger, train_writer, test_writer):

    model.eval()
    with torch.no_grad():
        data = data.to(device, dtype=torch.float)

        losses, intermediates = model(data, return_intermediates=True, writeout=False)
        utils.save_images(train_writer, model.step_counter, intermediates.input_image, intermediates.reconstruction,
            fname=os.path.join(args.figures_save, 'recon_epoch{}_idx{}_TRAIN_{:%Y_%m_%d_%H:%M}.jpg'.format(epoch, idx, datetime.datetime.now())))

        test_data = test_data.to(device, dtype=torch.float)
        losses, intermediates = model(test_data, return_intermediates=True, writeout=True)
        utils.save_images(test_writer, model.step_counter, intermediates.input_image, intermediates.reconstruction,
            fname=os.path.join(args.figures_save, 'recon_epoch{}_idx{}_TEST_{:%Y_%m_%d_%H:%M}.jpg'.format(epoch, idx, datetime.datetime.now())))

        compression_loss = losses['compression']
        epoch_test_loss.append(compression_loss.item())
        mean_test_loss = np.mean(epoch_test_loss)

        best_test_loss = utils.log(model, storage, epoch, idx, mean_test_loss, compression_loss.item(),
                                     best_test_loss, start_time, epoch_start_time,
                                     batch_size=data.shape[0], avg_bpp=test_bpp.mean().item(),header='[TEST]',
                                     logger=logger, writer=test_writer)

    return best_test_loss, epoch_test_loss


def train(args, model, train_loader, test_loader, device, logger, optimizers):

    start_time = time.time()
    test_loader_iter = iter(test_loader)
    current_D_steps, train_generator = 0, True
    best_loss, best_test_loss, mean_epoch_loss = np.inf, np.inf, np.inf
    train_writer = SummaryWriter(os.path.join(args.tensorboard_runs, 'train'))
    test_writer = SummaryWriter(os.path.join(args.tensorboard_runs, 'test'))
    storage, storage_test = model.storage_train, model.storage_test

    amortization_opt, hyperlatent_likelihood_opt = optimizers['amort'], optimizers['hyper']
    if model.use_discriminator is True:
        disc_opt = optimizers['disc']


    for epoch in trange(args.n_epochs, desc='Epoch'):

        epoch_loss, epoch_test_loss = [], []
        epoch_start_time = time.time()

        # if epoch > 0:
        #     ckpt_path = utils.save_model(model, optimizers, mean_epoch_loss, epoch, device, args=args, logger=logger)

        model.train()

        for idx, (data, bpp) in enumerate(tqdm(train_loader, desc='Train'), 0):

            data = data.to(device, dtype=torch.float)

            try:
                if model.use_discriminator is True:
                    # Train D for D_steps, then G, using distinct batches
                    losses = model(data, train_generator=train_generator)
                    compression_loss = losses['compression']
                    disc_loss = losses['disc']

                    if train_generator is True:
                        optimize_compression_loss(compression_loss, amortization_opt, hyperlatent_likelihood_opt)
                        train_generator = False
                    else:
                        optimize_loss(disc_loss, disc_opt)
                        current_D_steps += 1

                        if current_D_steps == args.discriminator_steps:
                            current_D_steps = 0
                            train_generator = True

                        continue
                else:
                    # Rate, distortion, perceptual only
                    losses = model(data, train_generator=True)
                    compression_loss = losses['compression']
                    optimize_compression_loss(compression_loss, amortization_opt, hyperlatent_likelihood_opt)

            except KeyboardInterrupt:
                # Note: saving not guaranteed!
                if model.step_counter > args.log_interval+1:
                    logger.warning('Exiting, saving ...')
                    ckpt_path = utils.save_model(model, optimizers, mean_epoch_loss, epoch, device, args=args, logger=logger)
                    return model, ckpt_path
                else:
                    return model, None

            # QAT: activate fake-quantize nodes after FP32 warmup
            qat_enabled = getattr(args, 'qat', False)
            if (qat_enabled
                    and not model._qat_active
                    and model.step_counter >= getattr(args, 'qat_warmup_steps', 50000)):
                logger.info(f'Step {model.step_counter}: activating QAT fake-quantize nodes...')
                _activate_qat_on_model(model, data, args, logger)

            # QAT: rebuild amortization optimizer with reduced LR after activation
            if model._qat_rebuild_optimizers:
                qat_lr = args.learning_rate * 0.1
                logger.info(f'Rebuilding amortization optimizer for QAT (lr={qat_lr})...')
                amort_params = itertools.chain.from_iterable(
                    [m.parameters() for m in model.amortization_models])
                amortization_opt = torch.optim.Adam(amort_params, lr=qat_lr)
                optimizers['amort'] = amortization_opt
                model._qat_rebuild_optimizers = False

            # QAT: freeze activation observers after freeze_steps to stabilise scales
            qat_freeze_steps = getattr(args, 'qat_freeze_steps', 70000)
            if (qat_enabled
                    and model._qat_active
                    and model.step_counter == qat_freeze_steps):
                logger.info(f'Step {model.step_counter}: freezing QAT activation observers.')
                _qat_nets = [model.Encoder, model.Generator, model.Hyperprior.analysis_net]
                if hasattr(model.Hyperprior, 'synthesis_mu'):
                    _qat_nets += [model.Hyperprior.synthesis_mu, model.Hyperprior.synthesis_std]
                elif hasattr(model.Hyperprior, 'synthesis_DLMM_params'):
                    _qat_nets.append(model.Hyperprior.synthesis_DLMM_params)
                for _net in _qat_nets:
                    torch.ao.quantization.disable_observer(_net)

            if model.step_counter % args.log_interval == 1:
                epoch_loss.append(compression_loss.item())
                mean_epoch_loss = np.mean(epoch_loss)

                best_loss = utils.log(model, storage, epoch, idx, mean_epoch_loss, compression_loss.item(),
                                best_loss, start_time, epoch_start_time, batch_size=data.shape[0],
                                avg_bpp=bpp.mean().item(), logger=logger, writer=train_writer)
                try:
                    test_data, test_bpp = next(test_loader_iter)
                except StopIteration:
                    test_loader_iter = iter(test_loader)
                    test_data, test_bpp = next(test_loader_iter)

                best_test_loss, epoch_test_loss = test(args, model, epoch, idx, data, test_data, test_bpp, device, epoch_test_loss, storage_test,
                     best_test_loss, start_time, epoch_start_time, logger, train_writer, test_writer)

                with open(os.path.join(args.storage_save, 'storage_{}_tmp.pkl'.format(args.name)), 'wb') as handle:
                    pickle.dump(storage, handle, protocol=pickle.HIGHEST_PROTOCOL)

                model.train()

                # LR scheduling
                utils.update_lr(args, amortization_opt, model.step_counter, logger)
                utils.update_lr(args, hyperlatent_likelihood_opt, model.step_counter, logger)
                if model.use_discriminator is True:
                    utils.update_lr(args, disc_opt, model.step_counter, logger)

                if model.step_counter > args.n_steps:
                    logger.info('Reached step limit [args.n_steps = {}]'.format(args.n_steps))
                    break

            if (idx % args.save_interval == 1) and (idx > args.save_interval):
                ckpt_path = utils.save_model(model, optimizers, mean_epoch_loss, epoch, device, args=args, logger=logger)

        # End epoch
        mean_epoch_loss = np.mean(epoch_loss)
        mean_epoch_test_loss = np.mean(epoch_test_loss)

        logger.info('===>> Epoch {} | Mean train loss: {:.3f} | Mean test loss: {:.3f}'.format(epoch,
            mean_epoch_loss, mean_epoch_test_loss))

        if model.step_counter > args.n_steps:
            break

    with open(os.path.join(args.storage_save, 'storage_{}_{:%Y_%m_%d_%H:%M:%S}.pkl'.format(args.name, datetime.datetime.now())), 'wb') as handle:
        pickle.dump(storage, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ckpt_path = utils.save_model(model, optimizers, mean_epoch_loss, epoch, device, args=args, logger=logger)
    args.ckpt = ckpt_path
    logger.info("Training complete. Time elapsed: {:.3f} s. Number of steps: {}".format((time.time()-start_time), model.step_counter))

    return model, ckpt_path


if __name__ == '__main__':

    description = "Learnable generative compression."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General options - see `default_config.py` for full options
    general = parser.add_argument_group('General options')
    general.add_argument("-n", "--name", default=None, help="Identifier for checkpoints and metrics.")
    general.add_argument("-mt", "--model_type", required=True, choices=(ModelTypes.COMPRESSION, ModelTypes.COMPRESSION_GAN),
        help="Type of model - with or without GAN component")
    general.add_argument("-regime", "--regime", choices=('low','med','high'), default='low', help="Set target bit rate - Low (0.14), Med (0.30), High (0.45)")
    general.add_argument("-gpu", "--gpu", type=int, default=0, help="GPU ID.")
    general.add_argument("-log_intv", "--log_interval", type=int, default=hific_args.log_interval, help="Number of steps between logs.")
    general.add_argument("-save_intv", "--save_interval", type=int, default=hific_args.save_interval, help="Number of steps between checkpoints.")
    general.add_argument("-multigpu", "--multigpu", help="Toggle data parallel capability using torch DataParallel", action="store_true")
    general.add_argument("-norm", "--normalize_input_image", help="Normalize input images to [-1,1]", action="store_true")
    general.add_argument('-bs', '--batch_size', type=int, default=hific_args.batch_size, help='input batch size for training')
    general.add_argument('--save', type=str, default='experiments', help='Parent directory for stored information (checkpoints, logs, etc.)')
    general.add_argument("-lt", "--likelihood_type", choices=('gaussian', 'logistic'), default='gaussian', help="Likelihood model for latents.")
    general.add_argument("-force_gpu", "--force_set_gpu", help="Set GPU to given ID", action="store_true")
    general.add_argument("-LMM", "--use_latent_mixture_model", help="Use latent mixture model as latent entropy model.", action="store_true")

    # Optimization-related options
    optim_args = parser.add_argument_group("Optimization-related options")
    optim_args.add_argument('-steps', '--n_steps', type=float, default=hific_args.n_steps,
        help="Number of gradient steps. Optimization stops at the earlier of n_steps/n_epochs.")
    optim_args.add_argument('-epochs', '--n_epochs', type=int, default=hific_args.n_epochs,
        help="Number of passes over training dataset. Optimization stops at the earlier of n_steps/n_epochs.")
    optim_args.add_argument("-lr", "--learning_rate", type=float, default=hific_args.learning_rate, help="Optimizer learning rate.")
    optim_args.add_argument("-wd", "--weight_decay", type=float, default=hific_args.weight_decay, help="Coefficient of L2 regularization.")

    # Architecture-related options
    arch_args = parser.add_argument_group("Architecture-related options")
    arch_args.add_argument('-lc', '--latent_channels', type=int, default=hific_args.latent_channels,
        help="Latent channels of bottleneck nominally compressible representation.")
    arch_args.add_argument('-nrb', '--n_residual_blocks', type=int, default=hific_args.n_residual_blocks,
        help="Number of residual blocks to use in Generator.")

    # Warmstart adversarial training from autoencoder/hyperprior
    warmstart_args = parser.add_argument_group("Warmstart options")
    warmstart_args.add_argument("-warmstart", "--warmstart", help="Warmstart adversarial training from autoencoder + hyperprior ckpt.", action="store_true")
    warmstart_args.add_argument("-ckpt", "--warmstart_ckpt", default=None, help="Path to autoencoder + hyperprior ckpt.")

    # QAT options
    qat_args = parser.add_argument_group("QAT (Quantization-Aware Training) options")
    qat_args.add_argument("--qat", action="store_true",
        help="Enable Quantization-Aware Training. Inserts fake-quantize nodes into "
             "Encoder, Generator, and Hyperprior nets after --qat_warmup_steps FP32 steps.")
    qat_args.add_argument("--qat_warmup_steps", type=int,
        default=hific_args.qat_warmup_steps,
        help="FP32 gradient steps before QAT fake-quantize nodes activate.")
    qat_args.add_argument("--qat_freeze_steps", type=int,
        default=hific_args.qat_freeze_steps,
        help="Training step at which activation observers are frozen.")
    qat_args.add_argument("--qat_backend", type=str, default=hific_args.qat_backend,
        choices=['x86', 'qnnpack'],
        help="Quantization backend: x86 for Intel/AMD CPUs, qnnpack for ARM.")

    cmd_args = parser.parse_args()

    if (cmd_args.gpu != 0) or (cmd_args.force_set_gpu is True):
        torch.cuda.set_device(cmd_args.gpu)

    if cmd_args.model_type == ModelTypes.COMPRESSION:
        args = mse_lpips_args
    elif cmd_args.model_type == ModelTypes.COMPRESSION_GAN:
        args = hific_args

    start_time = time.time()
    device = utils.get_device()

    # Override default arguments from config file with provided command line arguments
    dictify = lambda x: dict((n, getattr(x, n)) for n in dir(x) if not (n.startswith('__') or 'logger' in n))
    args_d, cmd_args_d = dictify(args), vars(cmd_args)
    args_d.update(cmd_args_d)
    args = utils.Struct(**args_d)
    args = utils.setup_generic_signature(args, special_info=args.model_type)
    args.target_rate = args.target_rate_map[args.regime]
    args.lambda_A = args.lambda_A_map[args.regime]
    args.n_steps = int(args.n_steps)

    storage = defaultdict(list)
    storage_test = defaultdict(list)
    logger = utils.logger_setup(logpath=os.path.join(args.snapshot, 'logs'), filepath=os.path.abspath(__file__))

    if args.warmstart is True:
        assert args.warmstart_ckpt is not None, 'Must provide checkpoint to previously trained AE/HP model.'
        logger.info('Warmstarting discriminator/generator from autoencoder/hyperprior model.')
        if args.model_type != ModelTypes.COMPRESSION_GAN:
            logger.warning('Should warmstart compression-gan model.')
        args, model, optimizers = utils.load_model(args.warmstart_ckpt, logger, device,
            model_type=args.model_type, current_args_d=dictify(args), strict=False, prediction=False, silent=True)
    else:
        model = create_model(args, device, logger, storage, storage_test)
        model = model.to(device)
        amortization_parameters = itertools.chain.from_iterable(
            [am.parameters() for am in model.amortization_models])

        hyperlatent_likelihood_parameters = model.Hyperprior.hyperlatent_likelihood.parameters()

        amortization_opt = torch.optim.Adam(amortization_parameters,
            lr=args.learning_rate)
        hyperlatent_likelihood_opt = torch.optim.Adam(hyperlatent_likelihood_parameters,
            lr=args.learning_rate)
        optimizers = dict(amort=amortization_opt, hyper=hyperlatent_likelihood_opt)

        if model.use_discriminator is True:
            discriminator_parameters = model.Discriminator.parameters()
            disc_opt = torch.optim.Adam(discriminator_parameters, lr=args.learning_rate)
            optimizers['disc'] = disc_opt

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1 and args.multigpu is True:
        # Not supported at this time
        raise NotImplementedError('MultiGPU not supported yet.')
        logger.info('Using {} GPUs.'.format(n_gpus))
        model = nn.DataParallel(model)

    logger.info('MODEL NAME: {}'.format(args.name))
    logger.info('MODEL TYPE: {}'.format(args.model_type))
    logger.info('MODEL MODE: {}'.format(args.model_mode))
    logger.info('BITRATE REGIME: {}'.format(args.regime))
    logger.info('SAVING LOGS/CHECKPOINTS/RECORDS TO {}'.format(args.snapshot))
    logger.info('USING DEVICE {}'.format(device))
    logger.info('USING GPU ID {}'.format(args.gpu))
    logger.info('USING DATASET: {}'.format(args.dataset))

    test_loader = datasets.get_dataloaders(args.dataset,
                                root=args.dataset_path,
                                batch_size=args.batch_size,
                                logger=logger,
                                mode='validation',
                                shuffle=True,
                                normalize=args.normalize_input_image)

    train_loader = datasets.get_dataloaders(args.dataset,
                                root=args.dataset_path,
                                batch_size=args.batch_size,
                                logger=logger,
                                mode='train',
                                shuffle=True,
                                normalize=args.normalize_input_image)

    args.n_data = len(train_loader.dataset)
    args.image_dims = train_loader.dataset.image_dims
    logger.info('Training elements: {}'.format(args.n_data))
    logger.info('Validation elements: {}'.format(len(test_loader.dataset)))
    logger.info('Input Dimensions: {}'.format(args.image_dims))
    # logger.info('Optimizers: {}'.format(optimizers))
    logger.info('Using device {}'.format(device))

    metadata = dict((n, getattr(args, n)) for n in dir(args) if not (n.startswith('__') or 'logger' in n))
    # logger.info(metadata)

    """
    Train
    """
    model, ckpt_path = train(args, model, train_loader, test_loader, device, logger, optimizers=optimizers)

    """
    TODO
    Generate metrics
    """
