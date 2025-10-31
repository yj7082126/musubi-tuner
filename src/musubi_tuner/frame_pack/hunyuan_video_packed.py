# original code: https://github.com/lllyasviel/FramePack
# original license: Apache-2.0

import glob
import math
import numbers
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import warnings
import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from musubi_tuner.modules.custom_offloading_utils import ModelOffloader
from musubi_tuner.utils.safetensors_utils import load_split_weights
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch, optimize_state_dict_with_fp8
from musubi_tuner.frame_pack.utils import repeat_to_batch_size
from accelerate import init_empty_weights

try:
    # raise NotImplementedError
    from xformers.ops import memory_efficient_attention as xformers_attn_func

    logging.info("Xformers is installed!")
except:
    logging.info("Xformers is not installed!")
    xformers_attn_func = None

try:
    # raise NotImplementedError
    from flash_attn import flash_attn_varlen_func, flash_attn_func

    logging.info("Flash Attn is installed!")
except:
    logging.info("Flash Attn is not installed!")
    flash_attn_varlen_func = None
    flash_attn_func = None

try:
    # raise NotImplementedError
    from sageattention import sageattn_varlen, sageattn

    logging.info("Sage Attn is installed!")
except:
    logging.info("Sage Attn is not installed!")
    sageattn_varlen = None
    sageattn = None


import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# region diffusers

# copied from diffusers with some modifications to minimize dependencies
# original code: https://github.com/huggingface/diffusers/
# original license: Apache-2.0

class HunyuanAttnCache(defaultdict):
    def __init__(self, *args, **kwargs):
        super().__init__(dict, *args, **kwargs)
        pass

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'HunyuanAttnCache' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value
    
    def clear(self):
        """Removes all items from the dictionary."""
        super().clear()
attn_cache = HunyuanAttnCache()

ACT2CLS = {
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}

def patchify(hidden_states, P=2, Q=2):
    hidden_states = einops.rearrange(hidden_states, "B C (H P) (W Q) -> B (H W) (C P Q)", P=P, Q=Q)
    return hidden_states

def unpatchify(hidden_states, height, width, P=2, Q=2):
    hidden_states = einops.rearrange(hidden_states, "B (H W) (C P Q) -> B C (H P) (W Q)", P=P, Q=Q, H=height//2, W=width//2)
    return hidden_states


def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACT2CLS:
        return ACT2CLS[act_fn]()
    else:
        raise ValueError(f"activation function {act_fn} not found in ACT2FN mapping {list(ACT2CLS.keys())}")


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


class FP32SiLU(nn.Module):
    r"""
    SiLU activation function with input upcasted to torch.float32.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.silu(inputs.float(), inplace=False).to(inputs.dtype)


class GELU(nn.Module):
    r"""
    GELU activation function with tanh approximation support with `approximate="tanh"`.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        # if gate.device.type == "mps" and is_torch_version("<", "2.0.0"):
        #     # fp16 gelu not supported on mps before torch 2.0
        #     return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate).to(dtype=gate.dtype)
        return F.gelu(gate, approximate=self.approximate)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class PixArtAlphaTextProjection(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        elif act_fn == "silu_fp32":
            self.act_1 = FP32SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LayerNormFramePack(nn.LayerNorm):
    # casting to dtype of input tensor is added
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps).to(x)


class FP32LayerNormFramePack(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        origin_dtype = x.dtype
        return torch.nn.functional.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


class RMSNormFramePack(nn.Module):
    r"""
    RMS Norm as introduced in https://arxiv.org/abs/1910.07467 by Zhang et al.

    Args:
        dim (`int`): Number of dimensions to use for `weights`. Only effective when `elementwise_affine` is True.
        eps (`float`): Small value to use when calculating the reciprocal of the square-root.
        elementwise_affine (`bool`, defaults to `True`):
            Boolean flag to denote if affine transformation should be applied.
        bias (`bool`, defaults to False): If also training the `bias` param.
    """

    def __init__(self, dim, eps: float, elementwise_affine: bool = True, bias: bool = False):
        super().__init__()

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        self.weight = None
        self.bias = None

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            if bias:
                self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is None:
            return hidden_states.to(input_dtype)

        return hidden_states.to(input_dtype) * self.weight.to(input_dtype)


class AdaLayerNormContinuousFramePack(nn.Module):
    r"""
    Adaptive normalization layer with a norm layer (layer_norm or rms_norm).

    Args:
        embedding_dim (`int`): Embedding dimension to use during projection.
        conditioning_embedding_dim (`int`): Dimension of the input condition.
        elementwise_affine (`bool`, defaults to `True`):
            Boolean flag to denote if affine transformation should be applied.
        eps (`float`, defaults to 1e-5): Epsilon factor.
        bias (`bias`, defaults to `True`): Boolean flag to denote if bias should be use.
        norm_type (`str`, defaults to `"layer_norm"`):
            Normalization layer to use. Values supported: "layer_norm", "rms_norm".
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # NOTE: It is a bit weird that the norm layer can be configured to have scale and shift parameters
        # because the output is immediately scaled and shifted by the projected conditioning embeddings.
        # Note that AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNormFramePack(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNormFramePack(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x, conditioning_embedding):
        emb = self.linear(self.silu(conditioning_embedding))
        scale, shift = emb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class LinearActivation(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True, activation: str = "silu"):
        super().__init__()

        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.activation = get_activation(activation)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        return self.activation(hidden_states)


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        # if activation_fn == "gelu":
        #     act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        # elif activation_fn == "geglu":
        #     act_fn = GEGLU(dim, inner_dim, bias=bias)
        # elif activation_fn == "geglu-approximate":
        #     act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        # elif activation_fn == "swiglu":
        #     act_fn = SwiGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "linear-silu":
            act_fn = LinearActivation(dim, inner_dim, bias=bias, activation="silu")
        else:
            raise ValueError(f"Unknown activation function: {activation_fn}")

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            # deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            # deprecate("scale", "1.0.0", deprecation_message)
            raise ValueError("scale is not supported in this version. Please remove it.")
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


# @maybe_allow_in_graph
class Attention(nn.Module):
    r"""
    Minimal copy of Attention class from diffusers.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        bias: bool = False,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        eps: float = 1e-5,
        processor: Optional[any] = None,
        out_dim: int = None,
        context_pre_only=None,
        pre_only=False,
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim  # if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only

        self.scale = dim_head**-0.5
        self.heads = out_dim // dim_head if out_dim is not None else heads

        self.added_kv_proj_dim = added_kv_proj_dim

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "rms_norm":
            self.norm_q = RMSNormFramePack(dim_head, eps=eps)
            self.norm_k = RMSNormFramePack(dim_head, eps=eps)
        else:
            raise ValueError(
                f"unknown qk_norm: {qk_norm}. Should be one of None, 'layer_norm', 'fp32_layer_norm', 'layer_norm_across_heads', 'rms_norm', 'rms_norm_across_heads', 'l2'."
            )

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)

        self.added_proj_bias = True  # added_proj_bias
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=True)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=True)
            if self.context_pre_only is not None:
                self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
        else:
            self.add_q_proj = None
            self.add_k_proj = None
            self.add_v_proj = None

        if not self.pre_only:
            self.to_out = nn.ModuleList([])
            self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=True))
            # self.to_out.append(nn.Dropout(dropout))
            self.to_out.append(nn.Identity())  # dropout=0.0
        else:
            self.to_out = None

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_add_out = nn.Linear(self.inner_dim, self.out_context_dim, bias=True)
        else:
            self.to_add_out = None

        if qk_norm is not None and added_kv_proj_dim is not None:
            if qk_norm == "rms_norm":
                self.norm_added_q = RMSNormFramePack(dim_head, eps=eps)
                self.norm_added_k = RMSNormFramePack(dim_head, eps=eps)
            else:
                raise ValueError(f"unknown qk_norm: {qk_norm}. Should be one of `None,'layer_norm','fp32_layer_norm','rms_norm'`")
        else:
            self.norm_added_q = None
            self.norm_added_k = None

        # set attention processor
        # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
        # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
        if processor is None:
            processor = AttnProcessor2_0()
        self.set_processor(processor)

    def set_processor(self, processor: any) -> None:
        self.processor = processor

    def get_processor(self) -> any:
        return self.processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *cross_attention_args, **cross_attention_kwargs,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            *cross_attention_args, **cross_attention_kwargs,
        )

    def prepare_attention_mask(
        self, attention_mask: torch.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ) -> torch.Tensor:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`torch.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `torch.Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
                padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
                #       we want to instead pad by (0, remaining_length), where remaining_length is:
                #       remaining_length: int = target_length - current_length
                # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0, output_size=attention_mask.shape[0] * head_size)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1, output_size=attention_mask.shape[1] * head_size)

        return attention_mask


class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        query_dtype = query.dtype  # store dtype before potentially deleting query

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        del query, key, value, attention_mask  # free memory

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query_dtype)  # use stored dtype

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states


# endregion diffusers


def pad_for_3d_conv(x, kernel_size):
    b, c, t, h, w = x.shape
    pt, ph, pw = kernel_size
    pad_t = (pt - (t % pt)) % pt
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")


def center_down_sample_3d(x, kernel_size):
    # pt, ph, pw = kernel_size
    # cp = (pt * ph * pw) // 2
    # xp = einops.rearrange(x, 'b c (t pt) (h ph) (w pw) -> (pt ph pw) b c t h w', pt=pt, ph=ph, pw=pw)
    # xc = xp[cp]
    # return xc
    return torch.nn.functional.avg_pool3d(x, kernel_size, stride=kernel_size)


def get_cu_seqlens(text_mask, img_len):
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device=text_mask.device)  # ensure device match

    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens


def apply_rotary_emb_transposed(x, freqs_cis):
    cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
    del freqs_cis
    x_real, x_imag = x.unflatten(-1, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    del x_real, x_imag
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


def attn_wscore_func(q, k, v, attn_bias=None):
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    # L, S = q.size(-2), k.size(-2)
    B, A, L, _ = q.shape
    S = k.size(2)
    scale_factor = 1 / math.sqrt(q.size(-1))
    if attn_bias is None:
        attn_bias = torch.zeros(B, A, L, S, dtype=q.dtype, device=q.device)
    else:
        if attn_bias.dim() == 3:
            attn_bias = attn_bias.unsqueeze(1)
        else:
            attn_bias = attn_bias

    attn_weight = q @ k.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(attn_weight.device)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_result = attn_weight @ v

    return attn_result.transpose(1, 2), attn_weight


def attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, attn_mask=None, attn_mode=None, split_attn=False):
    if cu_seqlens_q is None and cu_seqlens_kv is None and max_seqlen_q is None and max_seqlen_kv is None:
        if torch.isneginf(attn_mask).any():
            if attn_mode in ["sageattn", "flash"]:
                warnings.warn(f"You are using Attention Mask, but {attn_mode} does not support attention masking. Moving to normal SDPA")
            x = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask).transpose(
                1, 2
            )
            return x
        else:
            if attn_mode == "sageattn" or attn_mode is None and sageattn is not None:
                x = sageattn(q, k, v, tensor_layout="NHD")
                return x

            if attn_mode == "flash" or attn_mode is None and flash_attn_func is not None:
                x = flash_attn_func(q, k, v)
                return x

            if attn_mode == "xformers" or attn_mode is None and xformers_attn_func is not None:
                x = xformers_attn_func(q, k, v)
                return x
            x = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(
                1, 2
            )
            return x
    
    if split_attn:
        if attn_mode == "sageattn" or attn_mode is None and sageattn is not None:
            x = torch.empty_like(q)
            for i in range(q.size(0)):
                x[i : i + 1] = sageattn(q[i : i + 1], k[i : i + 1], v[i : i + 1], tensor_layout="NHD")
            return x

        if attn_mode == "flash" or attn_mode is None and flash_attn_func is not None:
            x = torch.empty_like(q)
            for i in range(q.size(0)):
                x[i : i + 1] = flash_attn_func(q[i : i + 1], k[i : i + 1], v[i : i + 1])
            return x

        if attn_mode == "xformers" or attn_mode is None and xformers_attn_func is not None:
            x = torch.empty_like(q)
            for i in range(q.size(0)):
                x[i : i + 1] = xformers_attn_func(q[i : i + 1], k[i : i + 1], v[i : i + 1])
            return x

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        x = torch.empty_like(q)
        for i in range(q.size(0)):
            x[i : i + 1] = torch.nn.functional.scaled_dot_product_attention(q[i : i + 1], k[i : i + 1], v[i : i + 1])
        x = x.transpose(1, 2)
        return x

    batch_size = q.shape[0]
    q = q.view(q.shape[0] * q.shape[1], *q.shape[2:])
    k = k.view(k.shape[0] * k.shape[1], *k.shape[2:])
    v = v.view(v.shape[0] * v.shape[1], *v.shape[2:])
    if attn_mode == "sageattn" or attn_mode is None and sageattn_varlen is not None:
        x = sageattn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
        del q, k, v  # free memory
    elif attn_mode == "flash" or attn_mode is None and flash_attn_varlen_func is not None:
        x = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
        del q, k, v  # free memory
    else:
        raise NotImplementedError("No Attn Installed or batch_size > 1 is not supported in this configuration. Try `--split_attn`.")
    x = x.view(batch_size, max_seqlen_q, *x.shape[1:])
    return x


class HunyuanAttnProcessorFlashAttnDouble:
    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states,
        attention_mask,
        seqlen_attention_mask,
        image_rotary_emb,
        attn_mode: Optional[str] = None,
        split_attn: Optional[bool] = False,
        *args, **kwargs
    ):
        cache_results = kwargs.get('cache_results', False)
        block_id = kwargs.get('block_id', '')
        timestep = kwargs.get('timestep', 0)
        print_results = cache_results and timestep >= 1000.0 and block_id.endswith('.0')
        if print_results:
            logging.info(f"[{block_id}]\n\t : hidden_states, encoder_hidden_states")

        cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv = seqlen_attention_mask
        if print_results:
            logging.info(f"[{block_id}]\n\tHs : {hidden_states} \n\tEhs : {encoder_hidden_states} \n\tIrs: {image_rotary_emb}")

        # Project image latents
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        del hidden_states  # free memory

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = apply_rotary_emb_transposed(query, image_rotary_emb)
        key = apply_rotary_emb_transposed(key, image_rotary_emb)
        del image_rotary_emb  # free memory

        # Project context (text/encoder) embeddings
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)
        txt_length = encoder_hidden_states.shape[1]  # store length before deleting
        del encoder_hidden_states  # free memory

        encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

        encoder_query = attn.norm_added_q(encoder_query)
        encoder_key = attn.norm_added_k(encoder_key)

        # Concatenate image and context q, k, v
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        del encoder_query, encoder_key, encoder_value  # free memory

        if print_results:
            logging.info(f"[{block_id}]\n\tQ : {query}\n\tK : {key}\n\tV : {value}\n\tAttn_mask : {attention_mask}")
            
        if cache_results:
            hidden_states_attn, attn_scores = attn_wscore_func(
                query, key, value, attn_bias=attention_mask
            )
            # keep a local tensor for potential in-forward use
            attn_for_use = attn_scores.mean(dim=1)
            # store a detached CPU copy for logging / caching so we don't keep the autograd graph
            try:
                attn_cache[block_id][timestep] = attn_for_use.detach().cpu().clone()
            except Exception:
                # fallback (shouldn't usually happen)
                attn_cache[block_id][timestep] = attn_for_use.detach().cpu()

            # If a connected_attn_map was provided by the caller, store the connected tensor
            connected_attn_map = kwargs.get('connected_attn_map', None)
            if connected_attn_map is not None:
                if block_id not in connected_attn_map:
                    connected_attn_map[block_id] = {}
                # Keep the connected (autograd) tensor on the original device for training use
                connected_attn_map[block_id] = attn_for_use

        else:
            if split_attn or attn_mode in ["sageattn", "flash"]:
                hidden_states_attn = attn_varlen_func(
                    query, key, value, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, attn_mask=attention_mask, attn_mode=attn_mode, split_attn=split_attn
                )
            else:
                hidden_states_attn, attn_scores = attn_wscore_func(
                    query, key, value, attn_bias=attention_mask
                )

        del query, key, value  # free memory
        if print_results:
            logging.info(f"[{block_id}]\n\thidden_states_attn : {hidden_states_attn}")

        hidden_states_attn = hidden_states_attn.flatten(-2)
        hidden_states, encoder_hidden_states = hidden_states_attn[:, :-txt_length], hidden_states_attn[:, -txt_length:]
        del hidden_states_attn  # free memory
        # Apply output projections
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)  # Dropout/Identity
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class HunyuanAttnProcessorFlashAttnSingle:
    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states,
        attention_mask,
        seqlen_attention_mask,
        image_rotary_emb,
        attn_mode: Optional[str] = None,
        split_attn: Optional[bool] = False,
        *args, **kwargs
    ):
        cache_results = kwargs.get('cache_results', False)
        block_id = kwargs.get('block_id', '')
        timestep = kwargs.get('timestep', 0)
        print_results = cache_results and timestep >= 1000.0 and block_id.endswith('.0')

        cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv = seqlen_attention_mask
        txt_length = encoder_hidden_states.shape[1]  # Store text length
        if print_results:
            logging.info(f"[{block_id}]\n\tHs : {hidden_states} \n\tEhs : {encoder_hidden_states} \n\tIrs: {image_rotary_emb}")

        # Concatenate image and context inputs
        hidden_states_cat = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        del hidden_states, encoder_hidden_states  # free memory

        # Project concatenated inputs
        query = attn.to_q(hidden_states_cat)
        key = attn.to_k(hidden_states_cat)
        value = attn.to_v(hidden_states_cat)
        del hidden_states_cat  # free memory

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = torch.cat([apply_rotary_emb_transposed(query[:, :-txt_length], image_rotary_emb), query[:, -txt_length:]], dim=1)
        key = torch.cat([apply_rotary_emb_transposed(key[:, :-txt_length], image_rotary_emb), key[:, -txt_length:]], dim=1)
        del image_rotary_emb  # free memory

        if cache_results:
            hidden_states, attn_scores = attn_wscore_func(
                query, key, value, attn_bias=attention_mask
            )
            # attn_cache[block_id][timestep] = attn_scores.clone().cpu()
            # For logging only: store a detached CPU copy so the global cache doesn't keep
            # autograd graph references which can cause "backward twice" errors later.
            attn_for_use = attn_scores.mean(dim=1)
            try:
                attn_cache[block_id][timestep] = attn_for_use.detach().cpu().clone()
            except Exception:
                # Fallback: at least move to CPU (no clone) if clone() fails for some dtype/device
                attn_cache[block_id][timestep] = attn_for_use.cpu()
            connected_attn_map = kwargs.get('connected_attn_map', None)
            if connected_attn_map is not None:
                if block_id not in connected_attn_map:
                    connected_attn_map[block_id] = {}
                connected_attn_map[block_id] = attn_for_use

        else:
            hidden_states = attn_varlen_func(
                query, key, value, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, attn_mask=attention_mask, attn_mode=attn_mode, split_attn=split_attn
            )

        del query, key, value  # free memory
        hidden_states = hidden_states.flatten(-2)

        hidden_states, encoder_hidden_states = hidden_states[:, :-txt_length], hidden_states[:, -txt_length:]

        return hidden_states, encoder_hidden_states


class CombinedTimestepGuidanceTextProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep, guidance, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))

        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))

        time_guidance_emb = timesteps_emb + guidance_emb

        pooled_projections = self.text_embedder(pooled_projection)
        conditioning = time_guidance_emb + pooled_projections

        return conditioning


class CombinedTimestepTextProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))

        pooled_projections = self.text_embedder(pooled_projection)

        conditioning = timesteps_emb + pooled_projections

        return conditioning


class HunyuanVideoAdaNorm(nn.Module):
    def __init__(self, in_features: int, out_features: Optional[int] = None) -> None:
        super().__init__()

        out_features = out_features or 2 * in_features
        self.linear = nn.Linear(in_features, out_features)
        self.nonlinearity = nn.SiLU()

    def forward(self, temb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        temb = self.linear(self.nonlinearity(temb))
        gate_msa, gate_mlp = temb.chunk(2, dim=-1)
        gate_msa, gate_mlp = gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)
        return gate_msa, gate_mlp


class HunyuanVideoIndividualTokenRefinerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = LayerNormFramePack(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
        )

        self.norm2 = LayerNormFramePack(hidden_size, elementwise_affine=True, eps=1e-6)
        self.ff = FeedForward(hidden_size, mult=mlp_width_ratio, activation_fn="linear-silu", dropout=mlp_drop_rate)

        self.norm_out = HunyuanVideoAdaNorm(hidden_size, 2 * hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        norm_hidden_states = self.norm1(hidden_states)

        # Self-attention
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
        )
        del norm_hidden_states  # free memory

        gate_msa, gate_mlp = self.norm_out(temb)
        hidden_states = hidden_states + attn_output * gate_msa
        del attn_output, gate_msa  # free memory

        ff_output = self.ff(self.norm2(hidden_states))
        hidden_states = hidden_states + ff_output * gate_mlp
        del ff_output, gate_mlp  # free memory

        return hidden_states


class HunyuanVideoIndividualTokenRefiner(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        self.refiner_blocks = nn.ModuleList(
            [
                HunyuanVideoIndividualTokenRefinerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    attention_bias=attention_bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self_attn_mask = None
        if attention_mask is not None:
            batch_size = attention_mask.shape[0]
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.to(hidden_states.device).bool()
            self_attn_mask_1 = attention_mask.view(batch_size, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
            self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
            self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()
            self_attn_mask[:, :, :, 0] = True

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, temb, self_attn_mask)

        return hidden_states


class HunyuanVideoTokenRefiner(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.time_text_embed = CombinedTimestepTextProjEmbeddings(embedding_dim=hidden_size, pooled_projection_dim=in_channels)
        self.proj_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.token_refiner = HunyuanVideoIndividualTokenRefiner(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            mlp_width_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            attention_bias=attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            pooled_projections = hidden_states.mean(dim=1)
        else:
            original_dtype = hidden_states.dtype
            mask_float = attention_mask.float().unsqueeze(-1)
            pooled_projections = (hidden_states * mask_float).sum(dim=1) / mask_float.sum(dim=1)
            pooled_projections = pooled_projections.to(original_dtype)

        temb = self.time_text_embed(timestep, pooled_projections)
        del pooled_projections  # free memory

        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.token_refiner(hidden_states, temb, attention_mask)
        del temb, attention_mask  # free memory

        return hidden_states


class HunyuanVideoRotaryPosEmbed(nn.Module):
    def __init__(self, rope_dim, theta):
        super().__init__()
        self.DT, self.DY, self.DX = rope_dim
        self.theta = theta
        self.h_w_scaling_factor = 1.0

    @torch.no_grad()
    def get_frequency(self, dim, pos):
        T, H, W = pos.shape
        freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device)[: (dim // 2)] / dim))
        freqs = torch.outer(freqs, pos.reshape(-1)).unflatten(-1, (T, H, W)).repeat_interleave(2, dim=0)
        return freqs.cos(), freqs.sin()

    @torch.no_grad()
    def forward_inner(self, frame_indices, height, width, device, 
                      start_height=0, start_width=0, step_H=1., step_W=1., 
                      num_H=None, num_W=None):
        if num_H is not None and num_W is not None:
            GT, GY, GX = torch.meshgrid(
                frame_indices.to(device=device, dtype=torch.float32),
                torch.linspace(start_height, height, num_H+1, device=device, dtype=torch.float32)[:-1] * self.h_w_scaling_factor,
                torch.linspace(start_width, width, num_W+1, device=device, dtype=torch.float32)[:-1] * self.h_w_scaling_factor,
                indexing="ij",
            )
        else:
            GT, GY, GX = torch.meshgrid(
                frame_indices.to(device=device, dtype=torch.float32),
                torch.arange(start_height, height, step_H, device=device, dtype=torch.float32) * self.h_w_scaling_factor,
                torch.arange(start_width, width, step_W, device=device, dtype=torch.float32) * self.h_w_scaling_factor,
                indexing="ij",
            )

        FCT, FST = self.get_frequency(self.DT, GT)
        del GT  # free memory
        FCY, FSY = self.get_frequency(self.DY, GY)
        del GY  # free memory
        FCX, FSX = self.get_frequency(self.DX, GX)
        del GX  # free memory

        result = torch.cat([FCT, FCY, FCX, FST, FSY, FSX], dim=0)
        del FCT, FCY, FCX, FST, FSY, FSX  # free memory

        # Return result already on the correct device
        return result  # Shape (2 * total_dim / 2, T, H, W) -> (total_dim, T, H, W)

    @torch.no_grad()
    def forward(self, frame_indices, height, width, device, 
                start_height=0, start_width=0, step_H=1., step_W=1., 
                num_H=None, num_W=None):
        frame_indices = frame_indices.unbind(0)
        results = [self.forward_inner(f, height, width, device, start_height, start_width, step_H, step_W, num_H, num_W) for f in frame_indices]
        results = torch.stack(results, dim=0)
        return results


class AdaLayerNormZero(nn.Module):
    def __init__(self, embedding_dim: int, norm_type="layer_norm", bias=True):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNormFramePack(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(
        self, x: torch.Tensor, emb: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = emb.unsqueeze(-2)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    def __init__(self, embedding_dim: int, norm_type="layer_norm", bias=True):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNormFramePack(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = emb.unsqueeze(-2)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=-1)
        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa


class AdaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNormFramePack(embedding_dim, eps, elementwise_affine, bias)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        emb = emb.unsqueeze(-2)
        emb = self.linear(self.silu(emb))
        scale, shift = emb.chunk(2, dim=-1)
        del emb  # free memory
        x = self.norm(x) * (1 + scale) + shift
        return x


class HunyuanVideoSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        qk_norm: str = "rms_norm",
        attn_mode: Optional[str] = None,
        split_attn: Optional[bool] = False,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim
        mlp_dim = int(hidden_size * mlp_ratio)
        self.attn_mode = attn_mode
        self.split_attn = split_attn

        # Attention layer (pre_only=True means no output projection in Attention module itself)
        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            bias=True,
            processor=HunyuanAttnProcessorFlashAttnSingle(),
            qk_norm=qk_norm,
            eps=1e-6,
            pre_only=True,  # Crucial: Attn processor will return raw attention output
        )

        self.norm = AdaLayerNormZeroSingle(hidden_size, norm_type="layer_norm")
        self.proj_mlp = nn.Linear(hidden_size, mlp_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(hidden_size + mlp_dim, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        seqlen_attention_mask: Optional[Tuple[torch.Tensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args, **kwargs
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        del encoder_hidden_states  # free memory

        residual = hidden_states

        # 1. Input normalization
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        norm_hidden_states, norm_encoder_hidden_states = (
            norm_hidden_states[:, :-text_seq_length, :],
            norm_hidden_states[:, -text_seq_length:, :],
        )

        # 2. Attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            seqlen_attention_mask=seqlen_attention_mask,
            image_rotary_emb=image_rotary_emb,
            attn_mode=self.attn_mode,
            split_attn=self.split_attn,
            *args, **kwargs
        )
        attn_output = torch.cat([attn_output, context_attn_output], dim=1)
        del norm_hidden_states, norm_encoder_hidden_states, context_attn_output  # free memory
        del image_rotary_emb

        # 3. Modulation and residual connection
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        del attn_output, mlp_hidden_states  # free memory
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = hidden_states + residual

        hidden_states, encoder_hidden_states = (
            hidden_states[:, :-text_seq_length, :],
            hidden_states[:, -text_seq_length:, :],
        )
        return hidden_states, encoder_hidden_states


class HunyuanVideoTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float,
        qk_norm: str = "rms_norm",
        attn_mode: Optional[str] = None,
        split_attn: Optional[bool] = False,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim
        self.attn_mode = attn_mode
        self.split_attn = split_attn

        self.norm1 = AdaLayerNormZero(hidden_size, norm_type="layer_norm")
        self.norm1_context = AdaLayerNormZero(hidden_size, norm_type="layer_norm")

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            added_kv_proj_dim=hidden_size,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            context_pre_only=False,
            bias=True,
            processor=HunyuanAttnProcessorFlashAttnDouble(),
            qk_norm=qk_norm,
            eps=1e-6,
        )

        self.norm2 = LayerNormFramePack(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")

        self.norm2_context = LayerNormFramePack(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        seqlen_attention_mask: Optional[Tuple[torch.Tensor]] = None,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Input normalization
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        # 2. Joint attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            seqlen_attention_mask=seqlen_attention_mask,
            image_rotary_emb=freqs_cis,
            attn_mode=self.attn_mode,
            split_attn=self.split_attn,
            *args, **kwargs
        )
        del norm_hidden_states, norm_encoder_hidden_states, freqs_cis  # free memory

        # 3. Modulation and residual connection
        hidden_states = hidden_states + attn_output * gate_msa
        del attn_output, gate_msa  # free memory
        encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa
        del context_attn_output, c_gate_msa  # free memory

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        del shift_mlp, scale_mlp  # free memory
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp
        del c_shift_mlp, c_scale_mlp  # free memory

        # 4. Feed-forward
        ff_output = self.ff(norm_hidden_states)
        del norm_hidden_states  # free memory
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        del norm_encoder_hidden_states  # free memory

        hidden_states = hidden_states + gate_mlp * ff_output
        del ff_output, gate_mlp  # free memory
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output
        del context_ff_output, c_gate_mlp  # free memory

        return hidden_states, encoder_hidden_states


class ClipVisionProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Linear(in_channels, out_channels * 3)
        self.down = nn.Linear(out_channels * 3, out_channels)

    def forward(self, x):
        projected_x = self.down(nn.functional.silu(self.up(x)))
        return projected_x


class HunyuanVideoPatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)


class HunyuanVideoPatchEmbedForCleanLatents(nn.Module):
    def __init__(self, inner_dim):
        super().__init__()
        self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))

    @torch.no_grad()
    def initialize_weight_from_another_conv3d(self, another_layer):
        weight = another_layer.weight.detach().clone()
        bias = another_layer.bias.detach().clone()

        sd = {
            "proj.weight": weight.clone(),
            "proj.bias": bias.clone(),
            "proj_2x.weight": einops.repeat(weight, "b c t h w -> b c (t tk) (h hk) (w wk)", tk=2, hk=2, wk=2) / 8.0,
            "proj_2x.bias": bias.clone(),
            "proj_4x.weight": einops.repeat(weight, "b c t h w -> b c (t tk) (h hk) (w wk)", tk=4, hk=4, wk=4) / 64.0,
            "proj_4x.bias": bias.clone(),
        }

        sd = {k: v.clone() for k, v in sd.items()}

        self.load_state_dict(sd)
        return


class HunyuanVideoTransformer3DModelPacked(nn.Module):  # (PreTrainedModelMixin, GenerationMixin,
    # ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    # @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 20,
        num_single_layers: int = 40,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        guidance_embeds: bool = True,
        text_embed_dim: int = 4096,
        pooled_projection_dim: int = 768,
        rope_theta: float = 256.0,
        rope_axes_dim: Tuple[int] = (16, 56, 56),
        has_image_proj=True,
        image_proj_dim=1152,
        has_clean_x_embedder=False,
        attn_mode: Optional[str] = None,
        split_attn: Optional[bool] = False,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels
        self.config_patch_size = patch_size
        self.config_patch_size_t = patch_size_t

        # 1. Latent and condition embedders
        self.x_embedder = HunyuanVideoPatchEmbed((patch_size_t, patch_size, patch_size), in_channels, inner_dim)
        self.context_embedder = HunyuanVideoTokenRefiner(
            text_embed_dim, num_attention_heads, attention_head_dim, num_layers=num_refiner_layers
        )
        self.time_text_embed = CombinedTimestepGuidanceTextProjEmbeddings(inner_dim, pooled_projection_dim)

        self.clean_x_embedder = None
        self.image_projection = None

        # 2. RoPE
        self.rope = HunyuanVideoRotaryPosEmbed(rope_axes_dim, rope_theta)

        # 3. Dual stream transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                HunyuanVideoTransformerBlock(
                    num_attention_heads,
                    attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    attn_mode=attn_mode,
                    split_attn=split_attn,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Single stream transformer blocks
        self.single_transformer_blocks = nn.ModuleList(
            [
                HunyuanVideoSingleTransformerBlock(
                    num_attention_heads,
                    attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    attn_mode=attn_mode,
                    split_attn=split_attn,
                )
                for _ in range(num_single_layers)
            ]
        )

        # 5. Output projection
        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, patch_size_t * patch_size * patch_size * out_channels)

        self.inner_dim = inner_dim
        self.use_gradient_checkpointing = False
        self.enable_teacache = False

        if has_image_proj:
            # self.install_image_projection(image_proj_dim)
            self.image_projection = ClipVisionProjection(in_channels=image_proj_dim, out_channels=self.inner_dim)
            # self.config["has_image_proj"] = True
            # self.config["image_proj_dim"] = in_channels

        # if has_clean_x_embedder:
        #     self.install_clean_x_embedder()
        self.clean_x_embedder = HunyuanVideoPatchEmbedForCleanLatents(self.inner_dim)
        # self.config["has_clean_x_embedder"] = True

        self.high_quality_fp32_output_for_inference = True  # False # change default to True

        # Block swapping attributes (initialized to None)
        self.blocks_to_swap = None
        self.offloader_double = None
        self.offloader_single = None

        # RoPE scaling
        self.rope_scaling_timestep_threshold: Optional[int] = None  # scale RoPE above this timestep
        self.rope_scaling_factor: float = 1.0  # RoPE scaling factor

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self):
        self.use_gradient_checkpointing = True
        logging.info("Gradient checkpointing enabled for HunyuanVideoTransformer3DModelPacked.")  # Logging

    def disable_gradient_checkpointing(self):
        self.use_gradient_checkpointing = False
        logging.info("Gradient checkpointing disabled for HunyuanVideoTransformer3DModelPacked.")  # Logging

    def initialize_teacache(self, enable_teacache=True, num_steps=25, rel_l1_thresh=0.15):
        self.enable_teacache = enable_teacache
        self.cnt = 0
        self.num_steps = num_steps
        self.rel_l1_thresh = rel_l1_thresh  # 0.1 for 1.6x speedup, 0.15 for 2.1x speedup
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.previous_residual = None
        self.teacache_rescale_func = np.poly1d([7.33226126e02, -4.01131952e02, 6.75869174e01, -3.14987800e00, 9.61237896e-02])
        if enable_teacache:
            logging.info(f"TeaCache enabled: num_steps={num_steps}, rel_l1_thresh={rel_l1_thresh}")
        else:
            logging.info("TeaCache disabled.")

    def gradient_checkpointing_method(self, block, *args, **kwargs):
        if self.use_gradient_checkpointing:
            result = torch.utils.checkpoint.checkpoint(block, use_reentrant=False, *args, **kwargs)
        else:
            result = block(*args, **kwargs)
        return result

    def enable_block_swap(self, num_blocks: int, device: torch.device, supports_backward: bool):
        self.blocks_to_swap = num_blocks
        self.num_double_blocks = len(self.transformer_blocks)
        self.num_single_blocks = len(self.single_transformer_blocks)
        double_blocks_to_swap = num_blocks // 2
        single_blocks_to_swap = (num_blocks - double_blocks_to_swap) * 2 + 1

        assert double_blocks_to_swap <= self.num_double_blocks - 1 and single_blocks_to_swap <= self.num_single_blocks - 1, (
            f"Cannot swap more than {self.num_double_blocks - 1} double blocks and {self.num_single_blocks - 1} single blocks. "
            f"Requested {double_blocks_to_swap} double blocks and {single_blocks_to_swap} single blocks."
        )

        self.offloader_double = ModelOffloader(
            "double",
            self.transformer_blocks,
            self.num_double_blocks,
            double_blocks_to_swap,
            supports_backward,
            device,
            # debug=True # Optional debugging
        )
        self.offloader_single = ModelOffloader(
            "single",
            self.single_transformer_blocks,
            self.num_single_blocks,
            single_blocks_to_swap,
            supports_backward,
            device,  # , debug=True
        )
        logging.info(
            f"HunyuanVideoTransformer3DModelPacked: Block swap enabled. Swapping {num_blocks} blocks, "
            + f"double blocks: {double_blocks_to_swap}, single blocks: {single_blocks_to_swap}, supports_backward: {supports_backward}."
        )

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap and self.blocks_to_swap > 0:
            self.offloader_double.set_forward_only(True)
            self.offloader_single.set_forward_only(True)
            self.prepare_block_swap_before_forward()
            logging.info(f"HunyuanVideoTransformer3DModelPacked: Block swap set to forward only.")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap and self.blocks_to_swap > 0:
            self.offloader_double.set_forward_only(False)
            self.offloader_single.set_forward_only(False)
            self.prepare_block_swap_before_forward()
            logging.info(f"HunyuanVideoTransformer3DModelPacked: Block swap set to forward and backward.")

    def move_to_device_except_swap_blocks(self, device: torch.device):
        # assume model is on cpu. do not move blocks to device to reduce temporary memory usage
        if self.blocks_to_swap:
            saved_double_blocks = self.transformer_blocks
            saved_single_blocks = self.single_transformer_blocks
            self.transformer_blocks = None
            self.single_transformer_blocks = None

        self.to(device)

        if self.blocks_to_swap:
            self.transformer_blocks = saved_double_blocks
            self.single_transformer_blocks = saved_single_blocks

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader_double.prepare_block_devices_before_forward(self.transformer_blocks)
        self.offloader_single.prepare_block_devices_before_forward(self.single_transformer_blocks)

    def enable_rope_scaling(self, timestep_threshold: Optional[int], rope_scaling_factor: float = 1.0):
        if timestep_threshold is not None and rope_scaling_factor > 0:
            self.rope_scaling_timestep_threshold = timestep_threshold
            self.rope_scaling_factor = rope_scaling_factor
            logger.info(f"RoPE scaling enabled: threshold={timestep_threshold}, scaling_factor={rope_scaling_factor}.")
        else:
            self.rope_scaling_timestep_threshold = None
            self.rope_scaling_factor = 1.0
            self.rope.h_w_scaling_factor = 1.0  # reset to default
            logger.info("RoPE scaling disabled.")

    def process_image_enbeddings(self, encoder_hidden_states, encoder_attention_mask, hidden_order_dict, 
                                 image_embeddings):
        
        noise_end = hidden_order_dict['noise'][-1][1]
        extra_encoder_hidden_states, extra_attention_masks = [], []
        for image_embedding in image_embeddings:
            extra_encoder_hidden_state = self.gradient_checkpointing_method(self.image_projection, image_embedding)
            extra_encoder_hidden_state = repeat_to_batch_size(extra_encoder_hidden_state, encoder_hidden_states.shape[0])
            extra_attention_mask = torch.ones(
                (encoder_hidden_states.shape[0], extra_encoder_hidden_state.shape[1]),
                dtype=encoder_attention_mask.dtype,
                device=encoder_attention_mask.device,
            )
            extra_encoder_hidden_states.append(extra_encoder_hidden_state)
            extra_attention_masks.append(extra_attention_mask)
            hidden_order_dict['image_embeddings'].append((noise_end, noise_end+extra_encoder_hidden_state.shape[1]))
            noise_end = noise_end+extra_encoder_hidden_state.shape[1]

        extra_encoder_hidden_states = torch.cat(extra_encoder_hidden_states, dim=1)
        extra_attention_masks = torch.cat(extra_attention_masks, dim=1)

        # must cat before (not after) encoder_hidden_states, due to attn masking
        encoder_hidden_states = torch.cat([extra_encoder_hidden_states, encoder_hidden_states], dim=1)
        encoder_attention_mask = torch.cat([extra_attention_masks, encoder_attention_mask], dim=1)

        del extra_encoder_hidden_states, extra_attention_mask  # free memory    
        return encoder_hidden_states, encoder_attention_mask, hidden_order_dict

    @torch.no_grad()
    def process_text_embeddings(self, encoder_hidden_states, encoder_attention_mask, hidden_order_dict, 
                                entity_inds=[]):
        ctrl_seq_len = hidden_order_dict['noise'][0][-1]
        if 'image_embeddings' in hidden_order_dict:
            ctrl_and_img_seq_len = hidden_order_dict['image_embeddings'][-1][-1]
        else:
            ctrl_and_img_seq_len = ctrl_seq_len

        if encoder_hidden_states.shape[0] == 1:
            # When batch size is 1, we do not need any masks or var-len funcs since cropping is mathematically same to what we want
            # If they are not same, then their impls are wrong. Ours are always the correct one.
            enc_seq_len = encoder_attention_mask.sum().item()
            encoder_hidden_states = encoder_hidden_states[:, :enc_seq_len]
            encoder_attention_mask = encoder_attention_mask[:, :enc_seq_len]
            seqlen_attention_mask = None, None, None, None
            if len(entity_inds) > 1:
                hidden_order_dict['text_embeddings'] = [(ctrl_and_img_seq_len+x[0], ctrl_and_img_seq_len+x[1]) for x in entity_inds]
            else:
                hidden_order_dict['text_embeddings'] = [(ctrl_and_img_seq_len, ctrl_seq_len+enc_seq_len)]
        else:
            # img_seq_len = hidden_states.shape[1]
            enc_seq_len = encoder_hidden_states.shape[1]

            cu_seqlens_q = get_cu_seqlens(encoder_attention_mask, ctrl_seq_len)
            cu_seqlens_kv = cu_seqlens_q
            max_seqlen_q = ctrl_seq_len + enc_seq_len
            max_seqlen_kv = max_seqlen_q

            seqlen_attention_mask = cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv
            del cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv  # free memory
            if len(entity_inds) > 1:
                hidden_order_dict['text_embeddings'] = [(ctrl_and_img_seq_len+x[0], ctrl_and_img_seq_len+x[1]) for x in entity_inds]
                hidden_order_dict['text_paddings'] = [(ctrl_and_img_seq_len+entity_inds[-1][1], ctrl_seq_len+enc_seq_len)]
            else:
                hidden_order_dict['text_embeddings'] = [(ctrl_and_img_seq_len, ctrl_seq_len+enc_seq_len)]
        return encoder_hidden_states, encoder_attention_mask, seqlen_attention_mask, hidden_order_dict

    def process_attention_mask(self, encoder_attention_mask, hidden_order_dict, 
                               use_attention_masking=[], entity_masks=None):
        ctrl_seq_len = hidden_order_dict['noise'][0][-1]
        hidden_attention_mask = torch.ones(
            (encoder_attention_mask.shape[0], ctrl_seq_len), 
            dtype=encoder_attention_mask.dtype,
            device=encoder_attention_mask.device,
        )
        attention_mask = torch.cat([hidden_attention_mask, encoder_attention_mask], dim=1)
        total_seq_len = attention_mask.shape[1]
        attention_mask = attention_mask.unsqueeze(1)
        attention_mask = attention_mask.repeat_interleave(
            total_seq_len, dim=1, 
            output_size=attention_mask.shape[1] * total_seq_len)
            
        if len(use_attention_masking) > 0:
            # Masking Logic
            ## No Cross between clean_latents
            if "no_cross_control_latents" in use_attention_masking:
                for i, (start_i, end_i) in enumerate(hidden_order_dict['clean_latents']):
                    for j, (start_j, end_j) in enumerate(hidden_order_dict['clean_latents']):
                        if i != j:
                            attention_mask[:, start_i:end_i, start_j:end_j] = False
            ## No Cross between text embedding and image embedding
            # if "no_cross_img_text" in use_attention_masking:
            #     (img_embed_start, img_embed_end) = hidden_order_dict['image_embeddings'][0]
            #     (txt_embed_start, txt_embed_end) = hidden_order_dict['text_embeddings'][0]
            #     attention_mask[:, img_embed_start:img_embed_end, txt_embed_start:txt_embed_end] = False
            #     attention_mask[:, txt_embed_start:txt_embed_end, img_embed_start:img_embed_end] = False

            ## Mask Entity Prompts by Mask
            if "mask_entities" in use_attention_masking:
                noise_start, noise_end = hidden_order_dict['noise'][0]
                if entity_masks is not None and len(hidden_order_dict['text_embeddings']) >= 1:
                    for i, (prompt_start, prompt_end) in enumerate(hidden_order_dict['text_embeddings'][1:]):
                        prompt_seq_len = prompt_end - prompt_start
                        entity_masks_patched = (torch.sum(patchify(entity_masks[:,:,i]), dim=-1) > 0) 
                        entity_masks_patched = entity_masks_patched.unsqueeze(1).repeat(1, prompt_seq_len, 1)
                        attention_mask[:, prompt_start:prompt_end, noise_start:noise_end] = entity_masks_patched
                        # image update with prompt
                        attention_mask[:, noise_start:noise_end, prompt_start:prompt_end] = entity_masks_patched.transpose(1, 2)

            ## Mask Control Regions by Mask
            if "mask_control" in use_attention_masking:
                noise_start, noise_end = hidden_order_dict['noise'][0]
                if entity_masks is not None and len(hidden_order_dict['clean_latents']) >= 1:
                    for i, (control_start, control_end) in enumerate(hidden_order_dict['clean_latents'][:]):
                        prompt_seq_len = control_end - control_start
                        entity_masks_patched = (torch.sum(patchify(entity_masks[:,:,i]), dim=-1) > 0) 
                        entity_masks_patched = entity_masks_patched.unsqueeze(1).repeat(1, prompt_seq_len, 1)
                        attention_mask[:, control_start:control_end, noise_start:noise_end] = entity_masks_patched
                        # image update with prompt
                        attention_mask[:, noise_start:noise_end, control_start:control_end] = entity_masks_patched.transpose(1, 2)

            ## Convert attention_mask type from bool to float
            attention_mask = attention_mask.float()
            attention_mask[attention_mask == 0] = float('-inf')
            attention_mask[attention_mask == 1] = 0
        
        return attention_mask

    def process_input_hidden_states(
        self,
        latents,
        hidden_order_dict,
        latent_indices=None,
        clean_latents=None,
        clean_latent_indices=None,
        clean_latent_bboxes=None, # [B, N, 4]
        clean_latents_2x=None,
        clean_latent_2x_indices=None,
        clean_latents_4x=None,
        clean_latent_4x_indices=None,
    ):
        hidden_states = self.gradient_checkpointing_method(self.x_embedder.proj, latents)
        B, C, T, H, W = hidden_states.shape

        if latent_indices is None:
            latent_indices = repeat_to_batch_size(torch.arange(0, T).unsqueeze(0), B)
        if latent_indices.shape[0] != B:
            latent_indices = repeat_to_batch_size(latent_indices, B)

        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_WH = hidden_states.shape[1]

        # logging.info(f"hidden_states shape: {B,C,T,H,W} -> {hidden_states.shape}, latent_indices: {latent_indices}")

        rope_freqs = self.rope(frame_indices=latent_indices, height=H, width=W, device=hidden_states.device)
        rope_freqs = rope_freqs.flatten(2).transpose(1, 2)
        hidden_order_dict['rope_freqs'].append(rope_freqs)
        if clean_latents is not None and clean_latent_indices is not None:
            processed_clean_latents, clean_latent_rope_freqs = [], []
            latent_n = len(clean_latents) if type(clean_latents) == list else clean_latents.shape[2]
            if type(clean_latents) != list:
                clean_latents = [clean_latents[:,:,[i]] for i in range(latent_n)]

            if clean_latent_indices.shape[1] != latent_n:
                clean_latent_indices = clean_latent_indices.repeat(1, latent_n)
            clean_latent_indices = [clean_latent_indices[:,[i]] for i in range(latent_n)]

            latent_counter = 0
            for i, clean_latent in enumerate(clean_latents):
                clean_latent = clean_latent.to(hidden_states)
                clean_latent = self.gradient_checkpointing_method(self.clean_x_embedder.proj, clean_latent)
                clean_latent_index = clean_latent_indices[i]

                if clean_latent.shape[0] != B:
                    clean_latent = repeat_to_batch_size(clean_latent, B)
                if clean_latent_index.shape[0] != B:
                    clean_latent_index = repeat_to_batch_size(clean_latent_index, B)
                
                _, _, _, clean_H, clean_W = clean_latent.shape

                if clean_latent_bboxes is not None:
                    clean_latent_bbox = clean_latent_bboxes[:,i]
                    if clean_latent_bbox.shape[0] != B:
                        clean_latent_bbox = repeat_to_batch_size(clean_latent_bbox, B)

                    clean_latent_rope_freq = []
                    for b in range(B):
                        cb = [
                            int(clean_latent_bbox[b,0]*W), int(clean_latent_bbox[b,1]*H), 
                            int(clean_latent_bbox[b,2]*W), int(clean_latent_bbox[b,3]*H)
                        ]
                        cb_rope_freq = self.rope(
                            frame_indices=clean_latent_index[[b]], 
                            height=cb[3], width=cb[2],
                            start_height=cb[1], start_width=cb[0],
                            num_H=clean_H,
                            num_W=clean_W,
                            device=clean_latent.device)
                        # logger.info(f"Clean Latent Rope Freq Shape: {cb_rope_freq.shape}")
                        clean_latent_rope_freq.append(cb_rope_freq)
                    clean_latent_rope_freq = torch.cat(clean_latent_rope_freq, dim=0)
                else:
                    clean_latent_rope_freq = self.rope(
                        frame_indices=clean_latent_index, 
                        height=clean_H, width=clean_W, 
                        device=clean_latent.device)
                # logger.info(f"Total Clean Latent Rope Freq Shape: {clean_latent_rope_freq.shape}")
                # if clean_latent_bbox is not None:
                #     logging.info(f"Cropping latents to {clean_latent_bbox}")
                #     clean_latent = clean_latent[:,:,:,
                #                                 int(clean_latent_bbox[1]//8):int(clean_latent_bbox[3]//8), 
                #                                 int(clean_latent_bbox[0]//8):int(clean_latent_bbox[2]//8)]
                #     clean_latent_rope_freq = clean_latent_rope_freq[:,:,:,
                #                                 int(clean_latent_bbox[1]//8):int(clean_latent_bbox[3]//8), 
                #                                 int(clean_latent_bbox[0]//8):int(clean_latent_bbox[2]//8)]

                clean_latent = clean_latent.flatten(2).transpose(1, 2)
                clean_latent_rope_freq = clean_latent_rope_freq.flatten(2).transpose(1, 2)
                hidden_order_dict['rope_freqs'].append(clean_latent_rope_freq)
                
                processed_clean_latents.append(clean_latent)
                clean_latent_rope_freqs.append(clean_latent_rope_freq)
                clean_WH = clean_latent.shape[1]
                hidden_order_dict['clean_latents'].append((latent_counter, latent_counter+clean_WH))
                latent_counter += clean_WH

            processed_clean_latents = torch.cat(processed_clean_latents, dim=1)
            clean_latent_rope_freqs = torch.cat(clean_latent_rope_freqs, dim=1)

            # logger.info(f"Clean Latent Rope Freq Shape: {clean_latent_rope_freq.shape}")
            # logger.info(f"Rope Freq Shape: {rope_freqs.shape}")
            hidden_states = torch.cat([processed_clean_latents, hidden_states], dim=1)
            rope_freqs = torch.cat([clean_latent_rope_freqs, rope_freqs], dim=1)
            
        if clean_latents_2x is not None and clean_latent_2x_indices is not None:
            clean_latents_2x = clean_latents_2x.to(hidden_states)
            clean_latents_2x = pad_for_3d_conv(clean_latents_2x, (2, 4, 4))
            clean_latents_2x = self.gradient_checkpointing_method(self.clean_x_embedder.proj_2x, clean_latents_2x)
            clean_latents_2x = clean_latents_2x.flatten(2).transpose(1, 2)

            clean_latent_2x_rope_freqs = self.rope(
                frame_indices=clean_latent_2x_indices, height=H, width=W, device=clean_latents_2x.device
            )
            clean_latent_2x_rope_freqs = pad_for_3d_conv(clean_latent_2x_rope_freqs, (2, 2, 2))
            clean_latent_2x_rope_freqs = center_down_sample_3d(clean_latent_2x_rope_freqs, (2, 2, 2))
            clean_latent_2x_rope_freqs = clean_latent_2x_rope_freqs.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([clean_latents_2x, hidden_states], dim=1)
            rope_freqs = torch.cat([clean_latent_2x_rope_freqs, rope_freqs], dim=1)

        if clean_latents_4x is not None and clean_latent_4x_indices is not None:
            clean_latents_4x = clean_latents_4x.to(hidden_states)
            clean_latents_4x = pad_for_3d_conv(clean_latents_4x, (4, 8, 8))
            clean_latents_4x = self.gradient_checkpointing_method(self.clean_x_embedder.proj_4x, clean_latents_4x)
            clean_latents_4x = clean_latents_4x.flatten(2).transpose(1, 2)

            clean_latent_4x_rope_freqs = self.rope(
                frame_indices=clean_latent_4x_indices, height=H, width=W, device=clean_latents_4x.device
            )
            clean_latent_4x_rope_freqs = pad_for_3d_conv(clean_latent_4x_rope_freqs, (4, 4, 4))
            clean_latent_4x_rope_freqs = center_down_sample_3d(clean_latent_4x_rope_freqs, (4, 4, 4))
            clean_latent_4x_rope_freqs = clean_latent_4x_rope_freqs.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([clean_latents_4x, hidden_states], dim=1)
            rope_freqs = torch.cat([clean_latent_4x_rope_freqs, rope_freqs], dim=1)

        # Indices corresponding to each latent type within attention mask
        clean_latents_start_ind = 0
        clean_latents_2x_start_ind = 0
        clean_latents_4x_start_ind = 0
        if clean_latents_4x is not None:
            clean_latents_2x_start_ind += clean_latents_4x.shape[1] 
            clean_latents_start_ind += clean_latents_4x.shape[1] 
        if clean_latents_2x is not None:
            clean_latents_start_ind += clean_latents_2x.shape[1] 

        if clean_latents is not None and clean_latent_indices is not None:
            hidden_order_dict['clean_latents'] = [(clean_latents_start_ind+x[0], clean_latents_start_ind+x[1]) for x in hidden_order_dict['clean_latents']]
        if clean_latents_2x is not None and clean_latent_2x_indices is not None:
            N = clean_latent_2x_indices.shape[-1]
            WH = clean_latents_2x.shape[1] // N
            hidden_order_dict['clean_latents_2x'] = [(clean_latents_2x_start_ind+i*WH, clean_latents_2x_start_ind+(i+1)*WH) for i in range(N)]
        if clean_latents_4x is not None and clean_latent_4x_indices is not None:
            N = clean_latent_4x_indices.shape[-1]
            WH = clean_latents_4x.shape[1] // N
            hidden_order_dict['clean_latents_4x'] = [(clean_latents_4x_start_ind+i*WH, clean_latents_4x_start_ind+(i+1)*WH) for i in range(N)]

        # noise_start_ind = clean_latents_start_ind + clean_latents.shape[1]
        hidden_order_dict['noise'] = [(hidden_states.shape[1]-hidden_WH, hidden_states.shape[1])]

        return hidden_states, rope_freqs, hidden_order_dict

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        pooled_projections,
        guidance,
        latent_indices=None,
        clean_latents=None,
        clean_latent_indices=None,
        clean_latent_bboxes=None,
        clean_latents_2x=None,
        clean_latent_2x_indices=None,
        clean_latents_4x=None,
        clean_latent_4x_indices=None,
        image_embeddings=None,
        attention_kwargs=None,
        return_dict=True,
        use_attention_masking=[],
        entity_inds=[],
        entity_masks=None,
        cache_results=False,
        cache_layers=[],
        return_connected_attn: bool = False,
        **kwargs
    ):
        hidden_order_dict = defaultdict(list)
        # If requested, collect connected (autograd-attached) attention tensors here for training use.
        connected_attn_map = {} if return_connected_attn else None
        if cache_results and timestep[0].item() >= 1000.0:
            logging.info("Activate Attention Caching for Exp.")
        if attention_kwargs is None:
            attention_kwargs = {}

        # RoPE scaling: must be done before processing hidden states
        if self.rope_scaling_timestep_threshold is not None:
            if timestep >= self.rope_scaling_timestep_threshold:
                self.rope.h_w_scaling_factor = self.rope_scaling_factor
            else:
                self.rope.h_w_scaling_factor = 1.0

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p, p_t = self.config_patch_size, self.config_patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p
        original_context_length = post_patch_num_frames * post_patch_height * post_patch_width

        hidden_states, rope_freqs, hidden_order_dict = self.process_input_hidden_states(
            hidden_states,
            hidden_order_dict,
            latent_indices,
            clean_latents,
            clean_latent_indices,
            clean_latent_bboxes,
            clean_latents_2x,
            clean_latent_2x_indices,
            clean_latents_4x,
            clean_latent_4x_indices,
        )
        del (
            latent_indices,
            clean_latents,
            clean_latent_indices,
            clean_latents_2x,
            clean_latent_2x_indices,
            clean_latents_4x,
            clean_latent_4x_indices,
        )  # free memory

        temb = self.gradient_checkpointing_method(self.time_text_embed, timestep, guidance, pooled_projections)
        encoder_hidden_states = self.gradient_checkpointing_method(
            self.context_embedder, encoder_hidden_states, timestep, encoder_attention_mask
        )

        if self.image_projection is not None:
            assert image_embeddings is not None, "You must use image embeddings!"
            encoder_hidden_states, encoder_attention_mask, hidden_order_dict = self.process_image_enbeddings(
                encoder_hidden_states, encoder_attention_mask, hidden_order_dict, image_embeddings
            )

        encoder_hidden_states, encoder_attention_mask, seqlen_attention_mask, hidden_order_dict = self.process_text_embeddings(
            encoder_hidden_states, encoder_attention_mask, hidden_order_dict, 
            entity_inds=entity_inds
        )

        attention_mask = self.process_attention_mask(
            encoder_attention_mask, hidden_order_dict, 
            use_attention_masking=use_attention_masking,
            entity_masks=entity_masks
        )
        attention_mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
        attn_cache['attn_mask'] = attention_mask.clone().cpu()
        attn_cache['attn_dict'] = hidden_order_dict
        del encoder_attention_mask

        if cache_results and timestep[0].item() >= 1000.0:
            logging.info(f"{timestep[0].item()}: hidden_states: {hidden_states}")
            logging.info(f"{timestep[0].item()}: encoder_hidden_states: {encoder_hidden_states}")
            logging.info(f"{timestep[0].item()}: temb: {temb}")
            logging.info(f"{timestep[0].item()}: attention_mask: {attention_mask}")
            logging.info(f"{timestep[0].item()}: rope_freqs: {rope_freqs}")
            logging.info(f"{timestep[0].item()}: {hidden_order_dict}")

        if self.enable_teacache:
            modulated_inp = self.transformer_blocks[0].norm1(hidden_states, emb=temb)[0]

            if self.cnt == 0 or self.cnt == self.num_steps - 1:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else:
                curr_rel_l1 = (
                    ((modulated_inp - self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean())
                    .cpu()
                    .item()
                )
                self.accumulated_rel_l1_distance += self.teacache_rescale_func(curr_rel_l1)
                should_calc = self.accumulated_rel_l1_distance >= self.rel_l1_thresh

                if should_calc:
                    self.accumulated_rel_l1_distance = 0

            self.previous_modulated_input = modulated_inp
            self.cnt += 1

            if self.cnt == self.num_steps:
                self.cnt = 0

            if not should_calc:
                hidden_states = hidden_states + self.previous_residual
            else:
                ori_hidden_states = hidden_states.clone()

                for block_id, block in enumerate(self.transformer_blocks):
                    hidden_states, encoder_hidden_states = self.gradient_checkpointing_method(
                        block, hidden_states, encoder_hidden_states, temb, attention_mask, seqlen_attention_mask, rope_freqs,
                        connected_attn_map=connected_attn_map if return_connected_attn else None
                    )

                for block_id, block in enumerate(self.single_transformer_blocks):
                    hidden_states, encoder_hidden_states = self.gradient_checkpointing_method(
                        block, hidden_states, encoder_hidden_states, temb, attention_mask, seqlen_attention_mask, rope_freqs,
                        connected_attn_map=connected_attn_map if return_connected_attn else None
                    )

                self.previous_residual = hidden_states - ori_hidden_states
                del ori_hidden_states  # free memory
        else:
            for block_id, block in enumerate(self.transformer_blocks):
                if self.blocks_to_swap:
                    self.offloader_double.wait_for_block(block_id)
                block_str = f"transformer_blocks.{block_id}"
                cache_results_layer = cache_results and block_str in cache_layers
                hidden_states, encoder_hidden_states = self.gradient_checkpointing_method(
                    block, hidden_states, encoder_hidden_states, temb, attention_mask, seqlen_attention_mask, rope_freqs,
                    cache_results=cache_results_layer,
                    block_id=block_str,
                    timestep=int(timestep[0].item()),
                    connected_attn_map=connected_attn_map if (return_connected_attn and cache_results_layer) else None,
                )

                if self.blocks_to_swap:
                    self.offloader_double.submit_move_blocks_forward(self.transformer_blocks, block_id)

            for block_id, block in enumerate(self.single_transformer_blocks):
                if self.blocks_to_swap:
                    self.offloader_single.wait_for_block(block_id)
                block_str = f"single_transformer_blocks.{block_id}"
                cache_results_layer = cache_results and block_str in cache_layers
                hidden_states, encoder_hidden_states = self.gradient_checkpointing_method(
                    block, hidden_states, encoder_hidden_states, temb, attention_mask, seqlen_attention_mask, rope_freqs,
                    cache_results=cache_results_layer,
                    block_id=block_str,
                    timestep=int(timestep[0].item()),
                    connected_attn_map=connected_attn_map if (return_connected_attn and cache_results_layer) else None,
                )

                if self.blocks_to_swap:
                    self.offloader_single.submit_move_blocks_forward(self.single_transformer_blocks, block_id)

        del attention_mask, seqlen_attention_mask, rope_freqs  # free memory
        del encoder_hidden_states  # free memory

        hidden_states = self.gradient_checkpointing_method(self.norm_out, hidden_states, temb)

        hidden_states = hidden_states[:, -original_context_length:, :]

        if self.high_quality_fp32_output_for_inference:
            hidden_states = hidden_states.to(dtype=torch.float32)
            if self.proj_out.weight.dtype != torch.float32:
                self.proj_out.to(dtype=torch.float32)

        hidden_states = self.gradient_checkpointing_method(self.proj_out, hidden_states)

        hidden_states = einops.rearrange(
            hidden_states,
            "b (t h w) (c pt ph pw) -> b c (t pt) (h ph) (w pw)",
            t=post_patch_num_frames,
            h=post_patch_height,
            w=post_patch_width,
            pt=p_t,
            ph=p,
            pw=p,
        )

        if return_connected_attn:
            clean_latent_inds = hidden_order_dict['clean_latents']
            noise_inds = hidden_order_dict['noise']
            block_id = cache_layers[0]
            attention_probs = connected_attn_map[block_id][:, noise_inds[0][0]:noise_inds[-1][1], :]

            attention_map = einops.rearrange(attention_probs, 'B (H W) D -> B H W D', H=post_patch_height, W=post_patch_width).permute(0,3,1,2)
            attention_map = attention_map[:,clean_latent_inds[0][0]:clean_latent_inds[-1][1],:,:].mean(axis=1).unsqueeze(1)
            attention_map = TF.resize(attention_map, size=(height, width))
            mn, mx = attention_map.amin(), attention_map.amax()
            denom = (mx - mn).clamp(min=1e-8)
            attention_map = ((attention_map - mn) / denom).clamp(0.0, 1.0)

        if return_dict:
            # return Transformer2DModelOutput(sample=hidden_states)
            ns = SimpleNamespace(sample=hidden_states)
            if return_connected_attn:
                ns.attention_map = attention_map
            return ns

        if return_connected_attn:
            return (hidden_states, attention_map)

        return (hidden_states,)

    def fp8_optimization(
        self, state_dict: dict[str, torch.Tensor], device: torch.device, move_to_device: bool, use_scaled_mm: bool = False
    ) -> dict[str, torch.Tensor]:  # Return type hint added
        """
        Optimize the model state_dict with fp8.

        Args:
            state_dict (dict[str, torch.Tensor]):
                The state_dict of the model.
            device (torch.device):
                The device to calculate the weight.
            move_to_device (bool):
                Whether to move the weight to the device after optimization.
            use_scaled_mm (bool):
                Whether to use scaled matrix multiplication for FP8.
        """
        TARGET_KEYS = ["transformer_blocks", "single_transformer_blocks"]
        EXCLUDE_KEYS = ["norm"]  # Exclude norm layers (e.g., LayerNorm, RMSNorm) from FP8

        # inplace optimization
        state_dict = optimize_state_dict_with_fp8(state_dict, device, TARGET_KEYS, EXCLUDE_KEYS, move_to_device=move_to_device)

        # apply monkey patching
        apply_fp8_monkey_patch(self, state_dict, use_scaled_mm=use_scaled_mm)

        return state_dict


def load_packed_model(
    device: Union[str, torch.device],
    dit_path: str,
    attn_mode: str,
    loading_device: Union[str, torch.device],
    fp8_scaled: bool = False,
    split_attn: bool = False,
    for_inference: bool = False,
    has_image_proj: bool = True
) -> HunyuanVideoTransformer3DModelPacked:
    # TODO support split_attn
    device = torch.device(device)
    loading_device = torch.device(loading_device)

    if os.path.isdir(dit_path):
        # we don't support from_pretrained for now, so loading safetensors directly
        safetensor_files = glob.glob(os.path.join(dit_path, "*.safetensors"))
        if len(safetensor_files) == 0:
            raise ValueError(f"Cannot find safetensors file in {dit_path}")
        # sort by name and take the first one
        safetensor_files.sort()
        dit_path = safetensor_files[0]

    with init_empty_weights():
        logger.info(f"Creating HunyuanVideoTransformer3DModelPacked")
        logger.info(f"Image Projection: {has_image_proj}")
        # import here to avoid circular import issues
        from musubi_tuner.frame_pack.hunyuan_video_packed_inference import HunyuanVideoTransformer3DModelPackedInference

        model_class = HunyuanVideoTransformer3DModelPackedInference if for_inference else HunyuanVideoTransformer3DModelPacked
        model = model_class(
            attention_head_dim=128,
            guidance_embeds=True,
            has_clean_x_embedder=True,
            has_image_proj=has_image_proj,
            image_proj_dim=1152,
            in_channels=16,
            mlp_ratio=4.0,
            num_attention_heads=24,
            num_layers=20,
            num_refiner_layers=2,
            num_single_layers=40,
            out_channels=16,
            patch_size=2,
            patch_size_t=1,
            pooled_projection_dim=768,
            qk_norm="rms_norm",
            rope_axes_dim=(16, 56, 56),
            rope_theta=256.0,
            text_embed_dim=4096,
            attn_mode=attn_mode,
            split_attn=split_attn,
        )

    # if fp8_scaled, load model weights to CPU to reduce VRAM usage. Otherwise, load to the specified device (CPU for block swap or CUDA for others)
    dit_loading_device = torch.device("cpu") if fp8_scaled else loading_device
    logger.info(f"Loading DiT model from {dit_path}, device={dit_loading_device}")

    # load model weights with the specified dtype or as is
    sd = load_split_weights(dit_path, device=dit_loading_device, disable_mmap=True)

    if fp8_scaled:
        # fp8 optimization: calculate on CUDA, move back to CPU if loading_device is CPU (block swap)
        logger.info(f"Optimizing model weights to fp8. This may take a while.")
        sd = model.fp8_optimization(sd, device, move_to_device=loading_device.type == "cpu")

        if loading_device.type != "cpu":
            # make sure all the model weights are on the loading_device
            logger.info(f"Moving weights to {loading_device}")
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)

    info = model.load_state_dict(sd, strict=False, assign=True)
    logger.info(f"Loaded DiT model from {dit_path}, info={info}")

    return model
