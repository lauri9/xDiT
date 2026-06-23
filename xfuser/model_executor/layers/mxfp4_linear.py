import torch
import torch.nn as nn
import math
try:
    import aiter
    from aiter.ops.shuffle import shuffle_weight
except ImportError:
    pass # Error will be thrown in base_model.py, if mxfp4 gemms are enabled but AITER is not available.
from typing import Optional

from xfuser.core.distributed.runtime_state import get_runtime_state
from xfuser.core.quant.activation_stats import maybe_log_activation_stats, should_log_activation_stats
from xfuser.core.quant.hadamard import (
    fold_weight_for_hadamard,
    get_hadamard_matrix_for_device,
    hadamard_rotate,
)
from xfuser.envs import environment_variables, parse_env_bool, parse_mxfp4_hadamard_block_r


@torch.library.custom_op("mylib::mxfp4_gemm", mutates_args=())
def _mxfp4_gemm(a: torch.Tensor, w_quant: torch.Tensor, w_scale: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    quant_func = aiter.get_hip_quant(aiter.QuantType.per_1x32)
    a_quant, a_scale = quant_func(a, shuffle=True)
    output = aiter.gemm_a4w4(a_quant, w_quant, a_scale, w_scale, bpreshuffle=True, bias=bias)
    return output

@_mxfp4_gemm.register_fake
def _(a: torch.Tensor, w_quant: torch.Tensor, w_scale: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Fake implementation for torch.compile shape inference
    """
    M, _ = a.shape
    N, _ = w_quant.shape
    
    # Return fake tensor with correct shape
    return torch.empty(M, N, dtype=a.dtype, device=a.device)


def _resolve_use_hadamard(explicit: Optional[bool]) -> bool:
    if explicit is not None:
        return explicit
    return parse_env_bool(environment_variables["AITER_MXFP4_HADAMARD"](), default=False)


def _resolve_hadamard_block_r(explicit: Optional[int]) -> int:
    if explicit is not None:
        return parse_mxfp4_hadamard_block_r(explicit)
    return parse_mxfp4_hadamard_block_r(environment_variables["AITER_MXFP4_BLOCK_R"]())


def _current_step() -> Optional[int]:
    runtime_state = get_runtime_state()
    step_counter = getattr(runtime_state, "step_counter", None)
    if step_counter is None:
        return None
    return int(step_counter)


class xFuserMXFP4Linear(nn.Module):
    """
    Custom Linear layer using MXFP4 GEMM operation
    
    Drop-in replacement for nn.Linear.
    """
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        *,
        use_hadamard: Optional[bool] = None,
        hadamard_block_r: Optional[int] = None,
        layer_name: str = "",
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_hadamard = _resolve_use_hadamard(use_hadamard)
        self.hadamard_block_r = _resolve_hadamard_block_r(hadamard_block_r)
        self.layer_name = layer_name

        if self.use_hadamard and in_features % self.hadamard_block_r != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by hadamard_block_r "
                f"({self.hadamard_block_r}) when Hadamard is enabled"
            )
        
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, **factory_kwargs)
            )
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        self.mm = self._run_mxfp4_gemm
    
    def reset_parameters(self) -> None:
        """Initialize weights using Kaiming uniform (same as nn.Linear)"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def load_and_quantize_weights(
        self, 
        weights: torch.Tensor, 
        bias: Optional[torch.Tensor] = None
    ) -> None:
        """
        Load pre-trained weights and quantize them.
        
        Args:
            weights: Full-precision weight tensor [out_features, in_features]
            bias: Optional bias tensor [out_features]
        """
        with torch.no_grad():
            # Temporarily restore weight parameter if it was deleted
            if self.weight is None:
                self.weight = nn.Parameter(
                    torch.empty_like(weights, device=weights.device, dtype=weights.dtype)
                )
            
            self.weight.data.copy_(weights.data)
            if bias is not None and self.bias is not None:
                self.bias.data.copy_(bias.data)
        
        self._quantize_weights()
    
    def _prepare_weight_for_quant(self, weight: torch.Tensor) -> torch.Tensor:
        if not self.use_hadamard:
            return weight
        return fold_weight_for_hadamard(weight, self.hadamard_block_r)

    def _quantize_weights(self) -> None:
        """
        Quantize weights to FP4 and register quantized tensors as buffers.
        
        This ensures proper device movement with .to(), .cuda(), CPU offload,
        and distributed training frameworks (FSDP, DDP).
        """
        if self.weight is None:
            raise RuntimeError(
                "Cannot quantize: weight parameter is None."
                "Call load_and_quantize_weights() or reset_parameters() first."
            )

        weight_to_quant = self._prepare_weight_for_quant(self.weight)
        quant_func = aiter.get_hip_quant(aiter.QuantType.per_1x32)
        weight_quant, weight_scale = quant_func(weight_to_quant, shuffle=True)
        weight_shuffle = shuffle_weight(weight_quant, layout=(16, 16))
        
        # Register quantized tensors as buffers for proper state management
        # persistent=True ensures they're saved in state_dict
        self.register_buffer('weight_shuffle', weight_shuffle, persistent=True)
        self.register_buffer('weight_scale', weight_scale, persistent=True)
        
        # Properly remove the original weight parameter to save memory
        # This maintains module structure while freeing memory
        delattr(self, 'weight')
        self.register_parameter('weight', None)

    def _run_mxfp4_gemm(self, a: torch.Tensor, w_quant: torch.Tensor, w_scale: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.ops.mylib.mxfp4_gemm(a, w_quant, w_scale, bias)

    def _maybe_rotate_activations(self, input_2d: torch.Tensor) -> torch.Tensor:
        if not self.use_hadamard:
            return input_2d
        R = get_hadamard_matrix_for_device(self.hadamard_block_r, input_2d.device)
        return hadamard_rotate(input_2d, R).contiguous()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using MXFP4 GEMM
        """

        if not hasattr(self, "weight_shuffle"):
            self._quantize_weights()

        # Save original shape
        original_shape = input.shape
        
        # Flatten all batch dimensions: [..., in_features] -> [M, in_features]
        input_2d = input.view(-1, self.in_features)

        if should_log_activation_stats():
            maybe_log_activation_stats(
                layer_name=self.layer_name,
                x=input_2d,
                group_size=32,
                phase="pre_hadamard",
                hadamard_enabled=self.use_hadamard,
                step=_current_step(),
            )

        input_2d = self._maybe_rotate_activations(input_2d)

        if should_log_activation_stats() and self.use_hadamard:
            maybe_log_activation_stats(
                layer_name=self.layer_name,
                x=input_2d,
                group_size=32,
                phase="post_hadamard",
                hadamard_enabled=True,
                step=_current_step(),
            )
        
        output = self.mm(
            input_2d,
            self.weight_shuffle,
            self.weight_scale,
            None
        )
        if self.bias is not None:
            output = output + self.bias
        
        # Reshape back to original batch dimensions
        # [M, N] -> [..., out_features]
        output = output.view(*original_shape[:-1], self.out_features)
        
        return output
    
    def extra_repr(self):
        """String representation (for print(model))"""
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'bias={self.bias is not None}, use_hadamard={self.use_hadamard}, '
            f'hadamard_block_r={self.hadamard_block_r}'
        )


class xFuserHybridMXFP4Linear(nn.Module):
    """
    Hybrid linear layer that switches per diffusion step between
    high precision (FP8-quantized nn.Linear path) and low precision (MXFP4 GEMM path).
    """

    def __init__(
        self,
        high_precision_linear: nn.Module,
        low_precision_linear: xFuserMXFP4Linear,
    ) -> None:
        super().__init__()
        self.high_precision_linear = high_precision_linear
        self.low_precision_linear = low_precision_linear

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        runtime_state = get_runtime_state()
        use_high_precision = getattr(runtime_state, "use_high_precision_gemm", True)
        if use_high_precision:
            return self.high_precision_linear(input)
        return self.low_precision_linear(input)

    def extra_repr(self):
        return "hybrid_gemm_schedule=True"
