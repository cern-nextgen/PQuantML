# requires CUDA as this test mainly tests whether moving Quantizers to CUDA devices work
import pytest
import torch

from pquant.core.torch.quantizer import Quantizer

CUDA_AVAILABLE = torch.cuda.is_available()


def make_default_quantizer(**overrides):
    """
    Default quantizer used similiarly in training loop.
    """
    kwargs = dict(
        k=1,
        i=4,
        f=7,
        overflow="SAT",
        round_mode="RND",
        is_heterogeneous=False,
        is_data=False,
        granularity="per_channel",
        hgq_gamma=0.0003,
        place="datalane",
        dynamic_data=True,
    )
    kwargs.update(overrides)
    return Quantizer(**kwargs)


def assert_all_params_on(module: Quantizer, device: str):
    """
    Assert that all registered quantizing variables (b, f, k, i) of quantizer are on device.
    """
    for name, param in module.named_parameters():
        if name in ("b", "f", "k", "i"):
            assert param.device.type == device, f"Parameter '{name}' is on {param.device}, expected {device}"


class TestQuantizerDevices:
    def setup_method(self):
        self.device = "cuda" if CUDA_AVAILABLE else "cpu"

    def test_initial_construction_device_choice(self):
        """
        Tests that quantizing variables (b, f, k, i) are on the right device
        """
        q = make_default_quantizer()
        expected = "cuda" if CUDA_AVAILABLE else "cpu"
        assert_all_params_on(q, expected)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA to test CUDA placement")
    def test_reconstruction(self):
        """
        Tests reinitialization of quantizer
        """
        q = make_default_quantizer()
        for _ in range(3):
            q = make_default_quantizer()
            assert_all_params_on(q, "cuda")

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA to test CUDA placement")
    def test_set_quantization_bits_preserves_cuda(self):
        """
        Tests if calling set_quantization_bits changes device
        """
        q = make_default_quantizer()
        # Ensure initial parameters are all on CUDA
        assert_all_params_on(q, "cuda")

        # Call the real implementation
        q.set_quantization_bits(i=2, f=9)

        # After the call, all registered parameters should still be on CUDA
        assert_all_params_on(q, "cuda")

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA to test CUDA placement")
    def test_apply_final_compression(self):
        """
        Tests if calling apply_final_compression changes device
        """
        q = make_default_quantizer()
        q.to(self.device)
        assert_all_params_on(q, "cuda")

        # This will call get_quantization_bits and then reassign i, f, b,
        # and final_compression_done.data.
        q.apply_final_compression()

        assert_all_params_on(q, "cuda")

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA to test CUDA placement")
    def test_quantized_forward_pass(self):
        """
        Tests whether the output of a forward pass through a quantized network is still on device.
        If any of the bit parameters ended up on CPU, this should raise
        a device mismatch error.
        """
        q = make_default_quantizer()
        q.to(self.device)
        assert_all_params_on(q, "cuda")

        q.set_quantization_bits(i=3, f=5)
        assert_all_params_on(q, "cuda")

        # dummy input on CUDA
        x = torch.randn(4, 8, device=torch.device("cuda"))

        out = q(x)
        assert out.device.type == "cuda", "Output tensor is not on CUDA as expected"
