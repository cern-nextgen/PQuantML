#!/bin/bash

pytest test_ap.py
KERAS_BACKEND="torch" pytest test_ap.py
pytest test_pdp.py
KERAS_BACKEND="torch" pytest test_pdp.py
pytest test_wanda.py
KERAS_BACKEND="torch" pytest test_wanda.py
pytest test_keras_compression_layers.py
DATA_FORMAT=channels_last pytest test_keras_compression_layers.py
KERAS_BACKEND="torch" pytest test_torch_compression_layers.py
KERAS_BACKEND=torch pytest test_torch_onnx_converter.py
pytest test_keras_onnx_converter.py
KERAS_BACKEND=torch pytest test_torch_alkaid_conversion.py
pytest test_keras_alkaid_conversion.py
KERAS_BACKEND=torch pytest test_hgq_torch.py
pytest test_hgq_keras.py
