import torch

import os
import time

from src.helpers import utils
from default_config import ModelModes

device = torch.device('cpu')
logger = utils.logger_setup(logpath=os.path.join("data/logs", f'logs_{time.time()}'), filepath=os.path.abspath(__file__))

def dequantize_checkpoint(checkpoint_path, output_path):
    # Load the quantized checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # Iterate through all parameters in the checkpoint
    for key, value in checkpoint.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                # Check if the tensor is quantized
                if isinstance(sub_value, torch.Tensor) and sub_value.is_quantized:
                    # Dequantize the tensor
                    value[sub_key] = sub_value.dequantize()
        elif isinstance(value, torch.Tensor) and value.is_quantized:
            # Dequantize the tensor directly
            checkpoint[key] = value.dequantize()

    torch.save(checkpoint, output_path)

checkpoint_path = "experiments/quantized_coffee_compression_gan_2025_01_06_08_22_epoch199_idx25800_2025_01_07_13_37.pt"
output_path = "experiments/dequantized_coffee_compression_gan_2025_01_06_08_22_epoch199_idx25800_2025_01_07_13_37.pt"
dequantize_checkpoint(checkpoint_path, output_path)

loaded_args, model, _ = utils.load_model(output_path, logger, device, model_mode=ModelModes.EVALUATION, current_args_d=None, prediction=True, strict=False, silent=True)