import ast
import asyncio
from datetime import timedelta
import gc
import importlib
import argparse
import math
import os
import pathlib
import re
import sys
import random
import time
import json
from multiprocessing import Value
from typing import Any, Dict, List, Optional
import accelerate
import numpy as np
from packaging.version import Version
from PIL import Image

import huggingface_hub
import toml

import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm
from accelerate.utils import set_seed
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs, PartialState
from safetensors.torch import load_file
import transformers
from diffusers.optimization import (
    SchedulerType as DiffusersSchedulerType,
    TYPE_TO_SCHEDULER_FUNCTION as DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION,
)
from transformers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION

from dataset import config_utils
from dataset.image_video_dataset import ARCHITECTURE_WAN, ARCHITECTURE_WAN_FULL
from hunyuan_model.models import load_transformer, get_rotary_pos_embed_by_shape, HYVideoDiffusionTransformer
import hunyuan_model.text_encoder as text_encoder_module
from hunyuan_model.vae import load_vae, VAE_VER
import hunyuan_model.vae as vae_module
import hv_train_network
from modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
import networks.lora as lora_module
from dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from hv_generate_video import save_images_grid, save_videos_grid, resize_image_to_bucket, encode_to_latents
from hv_train_network import (
    NetworkTrainer,
    load_prompts,
    clean_memory_on_device,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

import logging

from utils import huggingface_utils, model_utils, train_utils, sai_model_spec
from wan.configs import WAN_CONFIGS
from wan.modules.clip import CLIPModel
from wan.modules.t5 import T5EncoderModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class WanNetworkTrainer(NetworkTrainer):
    def __init__(self):
        # super().__init__()
        self.config = None
        self._i2v_training = False

    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_WAN

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_WAN_FULL

    def assert_model_specific_args(self, args):
        self.config = WAN_CONFIGS[args.task]
        self._i2v_training = "i2v" in args.task

    @property
    def i2v_training(self) -> bool:
        return self._i2v_training

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        config = self.config
        device = accelerator.device
        t5_path, clip_path, fp8_t5 = args.t5, args.clip, args.fp8_t5

        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        def encode_for_text_encoder(text_encoder):
            sample_prompts_te_outputs = {}  # (prompt) -> (embeds, mask)
            with accelerator.autocast(), torch.no_grad():
                for prompt_dict in prompts:
                    for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", self.config["sample_neg_prompt"])]:
                        if p is None:
                            continue
                        if p not in sample_prompts_te_outputs:
                            logger.info(f"cache Text Encoder outputs for prompt: {p}")

                            prompt_outputs = text_encoder([p], device)
                            sample_prompts_te_outputs[p] = prompt_outputs

            return sample_prompts_te_outputs

        # Load Text Encoder 1 and encode
        logger.info(f"loading T5: {t5_path}")
        t5 = T5EncoderModel(text_len=config.text_len, dtype=config.t5_dtype, device=device, weight_path=t5_path, fp8=fp8_t5)

        logger.info("encoding with Text Encoder 1")
        te_outputs_1 = encode_for_text_encoder(t5)
        del t5

        # load CLIP and encode image (for I2V training)
        sample_prompts_image_embs = {}
        for prompt_dict in prompts:
            if prompt_dict.get("image_path", None) is not None:
                sample_prompts_image_embs[prompt_dict["image_path"]] = None

        if len(sample_prompts_image_embs) > 0:
            logger.info(f"loading CLIP: {clip_path}")
            assert clip_path is not None, "CLIP path is required for I2V training / I2V学習にはCLIPのパスが必要です"
            clip = CLIPModel(dtype=config.clip_dtype, device=device, weight_path=clip_path)
            clip.model.to(device)

            logger.info(f"Encoding image to CLIP context")
            with accelerator.autocast(), torch.no_grad():
                for image_path in sample_prompts_image_embs:
                    logger.info(f"Encoding image: {image_path}")
                    img = Image.open(image_path).convert("RGB")
                    img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device)  # -1 to 1
                    clip_context = clip.visual([img[:, None, :, :]])
                    sample_prompts_image_embs[image_path] = clip_context

        # prepare sample parameters
        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()

            p = prompt_dict.get("prompt", "")
            prompt_dict_copy["t5_embeds"] = te_outputs_1[p][0]

            p = prompt_dict.get("negative_prompt", None)
            if p is not None:
                prompt_dict_copy["negative_t5_embeds"] = te_outputs_1[p][0]

            p = prompt_dict.get("image_path", None)
            if p is not None:
                prompt_dict_copy["clip_embeds"] = sample_prompts_image_embs[p]

            sample_parameters.append(prompt_dict_copy)

        clean_memory_on_device(accelerator.device)

        return sample_parameters

    def do_inference(
        self,
        accelerator,
        args,
        sample_parameter,
        timesteps,
        vae,
        dit_dtype,
        transformer,
        scheduler,
        width,
        height,
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        image_path=None,
    ):
        """architecture dependent inference"""
        device = accelerator.device

        # Calculate latent video length based on VAE version
        latent_video_length = (frame_count - 1) // self.config["vae_stride"][0] + 1

        # Get embeddings
        context = sample_parameter["t5_embeds"].to(device=device, dtype=dit_dtype)
        if do_classifier_free_guidance:
            context_null = sample_parameter["negative_t5_embeds"].to(device=device, dtype=dit_dtype)

        num_channels_latents = 16  # model.in_dim
        vae_scale_factor = self.config["vae_stride"][1]

        # Initialize latents
        lat_h = height // vae_scale_factor
        lat_w = width // vae_scale_factor
        shape_or_frame = (1, num_channels_latents, 1, lat_h, lat_w)
        latents = []
        for _ in range(latent_video_length):
            latents.append(torch.randn(shape_or_frame, generator=generator, device=device, dtype=dit_dtype))
        latents = torch.cat(latents, dim=2)

        if self.i2v_training:
            # Move VAE to the appropriate device for sampling
            vae.to(device)
            vae.eval()

            image = Image.open(image_path)
            image = resize_image_to_bucket(image, (width, height))  # returns a numpy array
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(1).float()  # C, 1, H, W
            image = image / 127.5 - 1  # -1 to 1

            # Create mask for the required number of frames
            msk = torch.ones(1, latent_video_length, lat_h, lat_w, device=self.device)
            msk[:, 1:] = 0
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
            msk = msk.transpose(1, 2)[0]

            with accelerator.autocast(), torch.no_grad():
                # Zero padding for the required number of frames only
                padding_frames = latent_video_length - 1  # The first frame is the input image
                image = torch.concat([image, torch.zeros(3, padding_frames, height, width)], dim=1).to(device=device)
                y = vae.encode([image])[0]

            y = y[:, :latent_video_length]  # may be not needed
            image_latents = torch.concat([msk, y])

            vae.to("cpu")
            clean_memory_on_device(device)
        else:
            image_latents = None

        """
        －－－－－－－－－－－－－－－－－－－－－－－－－－
        俺様用しおり
        　 ∧＿∧ 　　
        　（　´∀｀）＜　今日はここまで書いた
        －－－－－－－－－－－－－－－－－－－－－－－－－－
        """

        # Guidance scale
        guidance_expand = torch.tensor([guidance_scale * 1000.0], dtype=torch.float32, device=device).to(dit_dtype)

        # Get rotary positional embeddings
        freqs_cos, freqs_sin = get_rotary_pos_embed_by_shape(transformer, latents.shape[2:])
        freqs_cos = freqs_cos.to(device=device, dtype=dit_dtype)
        freqs_sin = freqs_sin.to(device=device, dtype=dit_dtype)

        # Wrap the inner loop with tqdm to track progress over timesteps
        prompt_idx = sample_parameter.get("enum", 0)
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps, desc=f"Sampling timesteps for prompt {prompt_idx+1}")):
                latents_input = scheduler.scale_model_input(latents, t)

                if do_classifier_free_guidance:
                    latents_input = torch.cat([latents_input, latents_input], dim=0)  # 2, C, F, H, W

                if image_latents is not None:
                    latents_image_input = (
                        image_latents if not do_classifier_free_guidance else torch.cat([image_latents, image_latents], dim=0)
                    )
                    latents_input = torch.cat([latents_input, latents_image_input], dim=1)  # 1 or 2, C*2, F, H, W

                noise_pred = transformer(
                    latents_input,
                    t.repeat(latents.shape[0]).to(device=device, dtype=dit_dtype),
                    text_states=prompt_embeds,
                    text_mask=prompt_mask,
                    text_states_2=prompt_embeds_2,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                    guidance=guidance_expand,
                    return_dict=True,
                )["x"]

                # perform classifier free guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

                # Compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Move VAE to the appropriate device for sampling
        vae.to(device)
        vae.eval()

        # Decode latents to video
        if hasattr(vae.config, "shift_factor") and vae.config.shift_factor:
            latents = latents / vae.config.scaling_factor + vae.config.shift_factor
        else:
            latents = latents / vae.config.scaling_factor

        latents = latents.to(device=device, dtype=vae.dtype)
        with torch.no_grad():
            video = vae.decode(latents, return_dict=False)[0]
        video = (video / 2 + 0.5).clamp(0, 1)
        video = video.cpu().float()

        return video

    def load_vae(self, vae_dtype: torch.dtype, vae_path: str):
        vae, _, s_ratio, t_ratio = load_vae(vae_dtype=vae_dtype, device="cpu", vae_path=vae_path)

        if args.vae_chunk_size is not None:
            vae.set_chunk_size_for_causal_conv_3d(args.vae_chunk_size)
            logger.info(f"Set chunk_size to {args.vae_chunk_size} for CausalConv3d in VAE")
        if args.vae_spatial_tile_sample_min_size is not None:
            vae.enable_spatial_tiling(True)
            vae.tile_sample_min_size = args.vae_spatial_tile_sample_min_size
            vae.tile_latent_min_size = args.vae_spatial_tile_sample_min_size // 8
        elif args.vae_tiling:
            vae.enable_spatial_tiling(True)

        return vae

    def load_transformer(
        self,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: torch.dtype,
    ):
        transformer = load_transformer(dit_path, attn_mode, split_attn, loading_device, dit_weight_dtype, args.dit_in_channels)

        if args.img_in_txt_in_offloading:
            logger.info("Enable offloading img_in and txt_in to CPU")
            transformer.enable_img_in_txt_in_offloading()

        return transformer

    def scale_shift_latents(self, latents):
        latents = latents * vae_module.SCALING_FACTOR
        return latents

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ):
        bsz = latents.shape[0]

        # I2V training
        if self.i2v_training:
            image_latents = torch.zeros_like(latents)
            image_latents[:, :, :1, :, :] = latents[:, :, :1, :, :]
            noisy_model_input = torch.cat([noisy_model_input, image_latents], dim=1)  # concat along channel dim

        # ensure guidance_scale in args is float
        guidance_vec = torch.full((bsz,), float(args.guidance_scale), device=accelerator.device)  # , dtype=dit_dtype)

        # ensure the hidden state will require grad
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            guidance_vec.requires_grad_(True)

        pos_emb_shape = latents.shape[1:]
        if pos_emb_shape not in self.pos_embed_cache:
            freqs_cos, freqs_sin = get_rotary_pos_embed_by_shape(transformer, latents.shape[2:])
            # freqs_cos = freqs_cos.to(device=accelerator.device, dtype=dit_dtype)
            # freqs_sin = freqs_sin.to(device=accelerator.device, dtype=dit_dtype)
            self.pos_embed_cache[pos_emb_shape] = (freqs_cos, freqs_sin)
        else:
            freqs_cos, freqs_sin = self.pos_embed_cache[pos_emb_shape]

        # call DiT
        latents = latents.to(device=accelerator.device, dtype=network_dtype)
        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)
        with accelerator.autocast():
            model_pred = transformer(
                noisy_model_input,
                timesteps,
                text_states=batch["llm"],
                text_mask=batch["llm_mask"],
                text_states_2=batch["clipL"],
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                guidance=guidance_vec,
                return_dict=False,
            )

        # flow matching loss
        target = noise - latents

        return model_pred, target

    # endregion model specific


def wan_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Wan2.1 specific parser setup"""
    parser.add_argument("--task", type=str, default="t2v-14B", choices=list(WAN_CONFIGS.keys()), help="The task to run.")
    parser.add_argument("--t5", type=str, default=None, required=True, help="text encoder (T5) checkpoint path")
    parser.add_argument("--fp8_t5", action="store_true", help="use fp8 for Text Encoder model")
    parser.add_argument(
        "--clip",
        type=str,
        default=None,
        help="text encoder (CLIP) checkpoint path, optional. If training I2V model, this is required",
    )
    parser.add_argument("--vae_cache_cpu", action="store_true", help="cache features in VAE on CPU")
    return parser


if __name__ == "__main__":
    parser = hv_train_network.setup_parser_common()
    parser = wan_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = torch.bfloat16  # Wan2.1 only supports bfloat16

    trainer = WanNetworkTrainer()
    trainer.train(args)
