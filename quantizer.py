import time
import os

import torch
import torch.quantization

from default_config import ModelModes
from src.helpers import utils

torch.backends.quantized.engine = 'qnnpack'

# Setup
device = torch.device('cpu')
logger = utils.logger_setup(logpath=os.path.join("data/logs", f'logs_{time.time()}'), filepath=os.path.abspath(__file__))
ck_path = "experiments/coffee_compression_gan_2025_01_06_08_22_epoch199_idx25800_2025_01_07_13_37.pt"

init_model = torch.load(ck_path, map_location=device)

# Load model
loaded_args, model, _ = utils.load_model(ck_path, logger, device, model_mode=ModelModes.EVALUATION, current_args_d=None, prediction=True, strict=False, silent=True)

# model.eval()

# qconfig = torch.quantization.QConfig(
#     activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8),
#     weight=torch.quantization.default_observer.with_args(dtype=torch.qint8)
# )
# model.qconfig = qconfig
# torch.quantization.prepare(model, inplace=True)

# quantized_model = torch.quantization.convert(model, inplace=True)

# Save model
save_dict = {   'model_state_dict': model.state_dict(),
                # 'compression_optimizer_state_dict': init_model['compression_optimizer_state_dict'],
                # 'hyperprior_optimizer_state_dict': init_model['hyperprior_optimizer_state_dict'],
                'epoch': 0,
                'steps': 0,
                'args': init_model['args'],
            }

torch.save(save_dict, f=os.path.join("experiments", f'stripped_{os.path.basename(ck_path)}'))
