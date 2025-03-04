import argparse
import os
import glob
from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from dataset import config_utils
from dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from PIL import Image

import logging

from dataset.image_video_dataset import ItemInfo, save_latent_cache_wan, ARCHITECTURE_WAN
from utils.model_utils import str_to_dtype
from wan.configs import wan_i2v_14B
from wan.modules.vae import WanVAE
from wan.modules.clip import CLIPModel
import cache_latents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(vae: WanVAE, clip: Optional[CLIPModel], batch: list[ItemInfo]):
    contents = torch.stack([torch.from_numpy(item.content) for item in batch])
    if len(contents.shape) == 4:
        contents = contents.unsqueeze(1)  # B, H, W, C -> B, F, H, W, C

    contents = contents.permute(0, 4, 1, 2, 3).contiguous()  # B, C, F, H, W
    contents = contents.to(vae.device, dtype=vae.dtype)
    contents = contents / 127.5 - 1.0  # normalize to [-1, 1]

    h, w = contents.shape[3], contents.shape[4]
    if h < 8 or w < 8:
        item = batch[0]  # other items should have the same size
        raise ValueError(f"Image or video size too small: {item.item_key} and {len(batch) - 1} more, size: {item.original_size}")

    # print(f"encode batch: {contents.shape}")
    with torch.no_grad():
        latent = vae.encode(contents).latent_dist.sample()

        if clip is not None:
            # extract first frame of contents
            images = contents[:, :, 0, :, :]  # B, C, H, W, non contiguous view is fine
            clip_context = clip.visual(images)
        else:
            clip_context = None

    # # debug: decode and save
    # with torch.no_grad():
    #     latent_to_decode = latent / vae.config.scaling_factor
    #     images = vae.decode(latent_to_decode, return_dict=False)[0]
    #     images = (images / 2 + 0.5).clamp(0, 1)
    #     images = images.cpu().float().numpy()
    #     images = (images * 255).astype(np.uint8)
    #     images = images.transpose(0, 2, 3, 4, 1)  # B, C, F, H, W -> B, F, H, W, C
    #     for b in range(images.shape[0]):
    #         for f in range(images.shape[1]):
    #             fln = os.path.splitext(os.path.basename(batch[b].item_key))[0]
    #             img = Image.fromarray(images[b, f])
    #             img.save(f"./logs/decode_{fln}_{b}_{f:03d}.jpg")

    for item, l in zip(batch, latent):
        # print(f"save latent cache: {item.latent_cache_path}, latent shape: {l.shape}")
        save_latent_cache_wan(item, l, clip_context)


def main(args):
    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_WAN)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images)
        return

    assert args.vae is not None, "vae checkpoint is required"

    vae_path = args.vae

    logger.info(f"Loading VAE model from {vae_path}")
    vae_dtype = torch.bfloat16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
    cache_device = torch.device("cpu") if args.vae_cache_cpu else None
    vae = WanVAE(vae_path=vae_path, device=device, dtype=vae_dtype, cache_device=cache_device)

    if args.clip is not None:
        clip_dtype = wan_i2v_14B.i2v_14B["clip_dtype"]
        clip = CLIPModel(dtype=clip_dtype, device=device, weight_path=args.clip)
    else:
        clip = None

    # Encode images
    def encode(one_batch: list[ItemInfo]):
        encode_and_save_batch(vae, clip, one_batch)

    cache_latents.encode_datasets(datasets, encode, args)


def setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--vae_cache_cpu", action="store_true", help="cache features in VAE on CPU")
    parser.add_argument(
        "--clip",
        type=str,
        default=None,
        help="text encoder (CLIP) checkpoint path, optional. If training I2V model, this is required",
    )
    return parser


if __name__ == "__main__":
    parser = cache_latents.setup_parser_common()
    parser = setup_parser(parser)

    args = parser.parse_args()
    main(args)
