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
pytest test_torch_onnx_converter.py
pytest test_keras_onnx_converter.py
