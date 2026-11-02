"""Compressed linear layer wrapper: decode -> upload -> compute -> cleanup."""

import torch
from ..utils.logging import get_logger
from ..utils.timing import Timer, TimingRecord
from .decoder import decode_artifact
from .tt_adapter import to_tt_weight, cleanup_tt_tensor

log = get_logger("compressed_linear")

_TTNN_AVAILABLE = False
try:
    import ttnn
    _TTNN_AVAILABLE = True
except ImportError:
    pass


class CompressedLinearRuntime:
    """Runtime wrapper for a compressed linear layer.

    On each forward call:
    1. Decode compressed artifact to CPU BF16
    2. Upload to TT device
    3. Execute ttnn.linear
    4. Optionally clean up temporary weight tensor
    """

    def __init__(
        self,
        artifact_path: str,
        device=None,
        bias=None,
        config=None,
        cache_decoded: bool = False,
        verify: bool = False,
        verbose: bool = True,
    ):
        self.artifact_path = artifact_path
        self.device = device
        self.bias = bias
        self.config = config or {}
        self.cache_decoded = cache_decoded
        self.verify = verify
        self.verbose = verbose
        self._cached_weight = None
        self._cached_tt_weight = None
        self.timing = TimingRecord()

    def forward(self, x, program_config=None, compute_kernel_config=None,
                memory_config=None, output_dtype=None):
        """Execute compressed linear: decode -> upload -> compute.

        Args:
            x: input tensor (TT tensor or torch tensor)
            program_config: optional ttnn program config
            compute_kernel_config: optional compute kernel config
            memory_config: optional output memory config
            output_dtype: optional output dtype

        Returns:
            output tensor from ttnn.linear (or torch.matmul fallback)
        """
        if self._cached_tt_weight is not None and self.cache_decoded:
            tt_weight = self._cached_tt_weight
        else:
            with Timer() as t_decode:
                cpu_weight = decode_artifact(
                    self.artifact_path,
                    verify=self.verify,
                    verbose=self.verbose,
                )
            self.timing.record("decode_ms", t_decode.elapsed_s)

            with Timer() as t_upload:
                tt_weight = to_tt_weight(
                    cpu_weight,
                    device=self.device,
                    verbose=self.verbose,
                )
            self.timing.record("upload_ms", t_upload.elapsed_s)

            if self.cache_decoded:
                self._cached_weight = cpu_weight
                self._cached_tt_weight = tt_weight

        with Timer() as t_op:
            if _TTNN_AVAILABLE and self.device is not None:
                kwargs = {}
                if program_config is not None:
                    kwargs["program_config"] = program_config
                if compute_kernel_config is not None:
                    kwargs["compute_kernel_config"] = compute_kernel_config
                if memory_config is not None:
                    kwargs["memory_config"] = memory_config
                if output_dtype is not None:
                    kwargs["dtype"] = output_dtype

                output = ttnn.linear(x, tt_weight, bias=self.bias, **kwargs)
            else:
                if isinstance(x, torch.Tensor):
                    w = self._cached_weight if self._cached_weight is not None else cpu_weight
                    output = torch.matmul(x.float(), w.t().float()).to(torch.bfloat16)
                else:
                    raise RuntimeError("ttnn not available and input is not a torch tensor")
        self.timing.record("tt_op_ms", t_op.elapsed_s)

        if not self.cache_decoded and _TTNN_AVAILABLE and self.device is not None:
            cleanup_tt_tensor(tt_weight, verbose=self.verbose)

        total = sum(self.timing.durations.get(k, 0) for k in ["decode_ms", "upload_ms", "tt_op_ms"])
        self.timing.record("total_layer_ms", total)

        return output

    def get_metrics(self) -> dict:
        """Return timing metrics for the last forward call."""
        return self.timing.summary()

    def cleanup(self):
        """Release any cached TT tensors."""
        if self._cached_tt_weight is not None:
            cleanup_tt_tensor(self._cached_tt_weight)
            self._cached_tt_weight = None
        self._cached_weight = None
