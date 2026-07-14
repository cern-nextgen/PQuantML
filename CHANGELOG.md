# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Packaging, tooling, and repository hygiene configuration for upcoming CI enforcement:
  Ruff, mypy, bandit, pytest and coverage sections in `pyproject.toml`.
- `docs` and `dev` optional-dependency extras, and `pytest-cov` in the `test` extra.
- `package-data` declaration so the `configs/*.yaml` files ship in the wheel.
- `CONTRIBUTING.md` and this changelog, with the changelog surfaced in the documentation.
- README sections for installation, backend selection (`KERAS_BACKEND`), quick start,
  contributing and license.
- `update_mask()` on the Torch PDP pruning layer, matching the Keras implementation.
- Tool caches and coverage artifacts to `.gitignore`.

### Changed

- Package discovery now uses `[tool.setuptools.packages.find]` instead of a hardcoded list,
  so all subpackages are included in the distribution.
- Documentation version is read from the installed package metadata instead of being hardcoded.
- Replaced an uninferable `TypeVar` with a `QuantBits` type alias in the Torch layers.

### Removed

- The unused `all` optional-dependency extra. Install backends explicitly, e.g.
  `pip install pquant-ml[test,torch]`.
- A stray `__init__.py` at the repository root.

### Fixed

- `add_compression_layers()` no longer forces CUDA, so PQuant works on CPU-only machines.
- `pAdam` and `pSGD` no longer raise `TypeError` when called with positional arguments.
- `PQActivation.get_config()` no longer calls a method that does not exist on `nn.Module`.
- `_weight`/`_bias` are now registered as `Parameter` rather than a plain `Tensor`.
- MDMM pruning model defaults now use enum members instead of raw strings.
- `tox.ini` referenced a non-existent `testing` extra.
- `.readthedocs.yaml` pointed at the wrong Sphinx configuration path.
- `docs/Makefile` referenced a leftover `HGQ` source path.
- Broken documentation links in the README.
- All mypy errors across `src` and `tests`.

### Security

- `load_torch_hgq_model()` now passes `weights_only=True` to `torch.load()`, preventing
  arbitrary code execution when loading an untrusted checkpoint (CWE-502).

## [0.0.6]

- Latest tagged release before the changelog was introduced.
