import os
import random

import keras
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def set_image_data_format():
    if "DATA_FORMAT" in os.environ:
        keras.backend.set_image_data_format(os.environ["DATA_FORMAT"])
    else:
        keras.backend.set_image_data_format("channels_first")


@pytest.fixture(scope="session", autouse=True, params=[42])
def set_random_seed(request):
    """Set random seeds for reproducibility"""

    seed = request.param
    np.random.seed(seed)
    random.seed(seed)
    backend = keras.backend.backend()
    match backend:
        case "tensorflow":
            import tensorflow as tf

            tf.random.set_seed(seed)
        case "torch":
            import torch

            torch.manual_seed(seed)
        case _:
            raise ValueError(f"Unknown backend: {backend}")


@pytest.fixture(scope="session", autouse=True)
def configure_backend():
    backend = keras.backend.backend()

    match backend:
        case "tensorflow":
            import tensorflow as tf

            # Use full float32 matmul precision to match numpy/onnxruntime. Without this, TF uses
            # TF32 on Ampere+ GPUs and the ~1e-3 relative error exceeds tight test tolerances
            # (e.g. the keras→ONNX parity tests). Mirrors the torch 'highest' setting below.
            tf.config.experimental.enable_tensor_float_32_execution(False)
        case "torch":
            import torch

            torch.set_float32_matmul_precision("highest")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch.set_default_device(device)
            torch.set_default_dtype(torch.float32)
        case _:
            raise ValueError(f"Unknown backend: {backend}")
