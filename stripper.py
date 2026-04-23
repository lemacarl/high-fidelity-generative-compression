import argparse
import time
import os

import torch

from default_config import ModelModes
from src.helpers import utils

torch.backends.quantized.engine = 'qnnpack'

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True, help='Path to checkpoint file')
args = parser.parse_args()

# Setup
device = torch.device('cpu')
logger = utils.logger_setup(logpath=os.path.join("data", f'logs_{time.time()}'), filepath=os.path.abspath(__file__))
ck_path = args.ckpt

init_model = torch.load(ck_path, map_location=device)

# Load model
loaded_args, model, _ = utils.load_model(ck_path, logger, device, model_mode=ModelModes.EVALUATION, current_args_d=None, prediction=True, strict=False, silent=True)

# Save model
save_dict = {   'model_state_dict': model.state_dict(),
                'epoch': 0,
                'steps': 0,
                'args': init_model['args'],
            }

torch.save(save_dict, f=os.path.join("experiments", f'stripped_{os.path.basename(ck_path)}'))
