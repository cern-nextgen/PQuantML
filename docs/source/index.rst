.. PQuantMLdocumentation master file, created by
   sphinx-quickstart on Mon Dec 8 16:28:11 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root toctree directive.

===========================
PQuantML
===========================

.. image:: https://img.shields.io/badge/license-Apache%202.0-green.svg
   :target: LICENSE
.. image:: https://github.com/calad0i/HGQ/actions/workflows/sphinx-build.yml/badge.svg
   :target: https://github.com/nroope/PQuant
.. image:: https://badge.fury.io/py/hgq.svg
   :target: https://pypi.org/project/pquant-ml/

Welcome to the documentation for PQuantML, a hardware-aware model compression framework supporting:

- Joint pruning + quantization
- Layer-wise precision configuration
- Flexible training pipelines
- PyTorch and TensorFlow backends
- Knowledge distillation
- HGQ library intergration
- Integration with hardware-friendly toolchains (e.g., hls4ml)

PQuantML enables efficient deployment of compact neural networks on resource-constrained hardware such as FPGAs and embedded accelerators.

The paper describing the framework is available at: `PQuantML: A Tool for End-to-End Hardware-aware Model Compression <https://arxiv.org/abs/2603.26595>`_.


.. rst-class:: light
.. image:: _static/overview_pquant_updated.png
   :alt: PQuantML-overview
   :width: 100%
   :align: center




Key Features
------------

- Joint Quantization + Pruning: Combine bit-width reduction with structured pruning.
- Flexible Precision Control: Per-layer and mixed-precision configuration.
- Hardware-Aware Objective: Include resource constraints (DSP, LUT, BRAM) in training.
- Simple API: Configure compression through a single YAML or Python object.
- PyTorch Integration: Works with custom training/validation loops.
- Export Support: Model conversion towards hardware toolchains.

Contents
=========================

.. toctree::
   :maxdepth: 2

   status
   install
   getting_started
   reference
   faq


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
