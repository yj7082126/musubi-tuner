from concurrent.futures import ThreadPoolExecutor
from curses import meta
import glob
import json
import math
import os
import random
import time
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from safetensors.torch import save_file, load_file
from PIL import Image
import cv2
import av

from musubi_tuner.utils import safetensors_utils
from musubi_tuner.utils.model_utils import dtype_to_str
from musubi_tuner.utils.bbox_utils import get_bbox_from_meta, get_facebbox_from_bbox

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]

try:
    import pillow_avif

    IMAGE_EXTENSIONS.extend([".avif", ".AVIF"])
except:
    pass

# JPEG-XL on Linux
try:
    from jxlpy import JXLImagePlugin

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])
except:
    pass

# JPEG-XL on Windows
try:
    import pillow_jxl

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])
except:
    pass

VIDEO_EXTENSIONS = [
    ".mp4",
    ".webm",
    ".avi",
    ".mkv",
    ".mov",
    ".flv",
    ".wmv",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".MP4",
    ".WEBM",
    ".AVI",
    ".MKV",
    ".MOV",
    ".FLV",
    ".WMV",
    ".M4V",
    ".MPG",
    ".MPEG",
]  # some of them are not tested

ARCHITECTURE_HUNYUAN_VIDEO = "hv"
ARCHITECTURE_HUNYUAN_VIDEO_FULL = "hunyuan_video"
ARCHITECTURE_WAN = "wan"
ARCHITECTURE_WAN_FULL = "wan"
ARCHITECTURE_FRAMEPACK = "fp"
ARCHITECTURE_FRAMEPACK_FULL = "framepack"


def glob_images(directory, base="*"):
    img_paths = []
    for ext in IMAGE_EXTENSIONS:
        if base == "*":
            img_paths.extend(glob.glob(os.path.join(glob.escape(directory), base + ext)))
        else:
            img_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
    img_paths = list(set(img_paths))  # remove duplicates
    img_paths.sort()
    return img_paths


def glob_videos(directory, base="*"):
    video_paths = []
    for ext in VIDEO_EXTENSIONS:
        if base == "*":
            video_paths.extend(glob.glob(os.path.join(glob.escape(directory), base + ext)))
        else:
            video_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
    video_paths = list(set(video_paths))  # remove duplicates
    video_paths.sort()
    return video_paths


def divisible_by(num: int, divisor: int) -> int:
    return num - num % divisor


def resize_image_to_bucket(image: Union[Image.Image, np.ndarray], bucket_reso: tuple[int, int]) -> np.ndarray:
    """
    Resize the image to the bucket resolution.

    bucket_reso: **(width, height)**
    """
    is_pil_image = isinstance(image, Image.Image)
    if is_pil_image:
        image_width, image_height = image.size
    else:
        image_height, image_width = image.shape[:2]

    if bucket_reso == (image_width, image_height):
        return np.array(image) if is_pil_image else image

    bucket_width, bucket_height = bucket_reso

    # resize the image to the bucket resolution to match the short side
    scale_width = bucket_width / image_width
    scale_height = bucket_height / image_height
    scale = max(scale_width, scale_height)
    image_width = int(image_width * scale + 0.5)
    image_height = int(image_height * scale + 0.5)

    if scale > 1:
        image = Image.fromarray(image) if not is_pil_image else image
        image = image.resize((image_width, image_height), Image.LANCZOS)
        image = np.array(image)
    else:
        image = np.array(image) if is_pil_image else image
        image = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_AREA)

    # crop the image to the bucket resolution
    crop_left = (image_width - bucket_width) // 2
    crop_top = (image_height - bucket_height) // 2
    image = image[crop_top : crop_top + bucket_height, crop_left : crop_left + bucket_width]
    return image


class ItemInfo:
    def __init__(
        self,
        item_key: str,
        caption: str,
        original_size: tuple[int, int],
        bucket_size: Optional[tuple[Any]] = None,
        frame_count: Optional[int] = None,
        content: Optional[np.ndarray] = None,
        latent_cache_path: Optional[str] = None,
    ) -> None:
        self.item_key = item_key
        self.caption = caption
        self.original_size = original_size
        self.bucket_size = bucket_size
        self.frame_count = frame_count
        self.content = content
        self.latent_cache_path = latent_cache_path
        self.text_encoder_output_cache_path: Optional[str] = None

        # np.ndarray for video, list[np.ndarray] for image with multiple controls
        self.control_content = None
        # np.ndarray for image embedding if the first control content is not same as the image embedding input
        self.embed_content: Optional[np.ndarray] = content
        # list[np.ndarray] for numeric bbox information for RoPE embedding
        self.clean_latent_bboxes: Optional[list] = None

        # FramePack architecture specific
        self.fp_latent_window_size: Optional[int] = None
        self.fp_1f_clean_indices: Optional[list[int]] = None  # indices of clean latents for 1f
        self.fp_1f_target_index: Optional[int] = None  # target index for 1f clean latents
        self.fp_1f_no_post: Optional[bool] = None  # whether to add zero values as clean latent post

    def __str__(self) -> str:
        return (
            f"ItemInfo(item_key={self.item_key}, caption={self.caption}, "
            + f"original_size={self.original_size}, bucket_size={self.bucket_size}, "
            + f"frame_count={self.frame_count}, latent_cache_path={self.latent_cache_path}, content={self.content.shape if self.content is not None else None})"
        )


# We use simple if-else approach to support multiple architectures.
# Maybe we can use a plugin system in the future.

# the keys of the dict are `<content_type>_FxHxW_<dtype>` for latents
# and `<content_type>_<dtype|mask>` for other tensors


def save_latent_cache(item_info: ItemInfo, latent: torch.Tensor):
    """HunyuanVideo architecture only. HunyuanVideo doesn't support I2V and control latents"""
    assert latent.dim() == 4, "latent should be 4D tensor (frame, channel, height, width)"

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu()}

    save_latent_cache_common(item_info, sd, ARCHITECTURE_HUNYUAN_VIDEO_FULL)


def save_latent_cache_wan(
    item_info: ItemInfo,
    latent: torch.Tensor,
    clip_embed: Optional[torch.Tensor],
    image_latent: Optional[torch.Tensor],
    control_latent: Optional[torch.Tensor],
    f_indices: Optional[list[int]] = None,
):
    """Wan architecture only"""
    assert latent.dim() == 4, "latent should be 4D tensor (frame, channel, height, width)"

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu()}

    if clip_embed is not None:
        sd[f"clip_{dtype_str}"] = clip_embed.detach().cpu()

    if image_latent is not None:
        sd[f"latents_image_{F}x{H}x{W}_{dtype_str}"] = image_latent.detach().cpu()

    if control_latent is not None:
        sd[f"latents_control_{F}x{H}x{W}_{dtype_str}"] = control_latent.detach().cpu()

    if f_indices is not None:
        dtype_str = dtype_to_str(torch.int32)
        sd[f"f_indices_{dtype_str}"] = torch.tensor(f_indices, dtype=torch.int32)

    save_latent_cache_common(item_info, sd, ARCHITECTURE_WAN_FULL)


def save_latent_cache_framepack(
    item_info: ItemInfo,
    latent: torch.Tensor,
    latent_indices: torch.Tensor,
    clean_latents: torch.Tensor,
    clean_latent_indices: torch.Tensor,
    clean_latents_2x: torch.Tensor,
    clean_latent_2x_indices: torch.Tensor,
    clean_latents_4x: torch.Tensor,
    clean_latent_4x_indices: torch.Tensor,
    image_embeddings: torch.Tensor,
    target_latent_masks: torch.Tensor,
    clean_latent_bboxes: torch.Tensor
):
    """FramePack architecture only"""
    assert latent.dim() == 4, "latent should be 4D tensor (frame, channel, height, width)"

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    # `latents_xxx` must have {F, H, W} suffix
    indices_dtype_str = dtype_to_str(latent_indices.dtype)
    sd[f"image_embeddings_{dtype_str}"] = image_embeddings.detach().cpu()  # image embeddings dtype is same as latents dtype
    sd[f"latent_indices_{indices_dtype_str}"] = latent_indices.detach().cpu()
    sd[f"clean_latent_indices_{indices_dtype_str}"] = clean_latent_indices.detach().cpu()
    sd[f"latents_clean_{F}x{H}x{W}_{dtype_str}"] = clean_latents.detach().cpu().contiguous()
    if clean_latent_2x_indices is not None:
        sd[f"clean_latent_2x_indices_{indices_dtype_str}"] = clean_latent_2x_indices.detach().cpu()
    if clean_latents_2x is not None:
        sd[f"latents_clean_2x_{F}x{H}x{W}_{dtype_str}"] = clean_latents_2x.detach().cpu().contiguous()
    if clean_latent_4x_indices is not None:
        sd[f"clean_latent_4x_indices_{indices_dtype_str}"] = clean_latent_4x_indices.detach().cpu()
    if clean_latents_4x is not None:
        sd[f"latents_clean_4x_{F}x{H}x{W}_{dtype_str}"] = clean_latents_4x.detach().cpu().contiguous()
    if target_latent_masks is not None:
        sd[f"target_latent_masks_{F}x{H}x{W}_{dtype_str}"] = target_latent_masks.detach().cpu().contiguous()
    if clean_latent_bboxes is not None:
        sd[f"clean_latent_bboxes_float32"] = clean_latent_bboxes.detach().cpu().float().contiguous()
    # for key, value in sd.items():
    #     print(f"{key}: {value.shape}")
    save_latent_cache_common(item_info, sd, ARCHITECTURE_FRAMEPACK_FULL)


def save_latent_cache_common(item_info: ItemInfo, sd: dict[str, torch.Tensor], arch_fullname: str):
    metadata = {
        "architecture": arch_fullname,
        "width": f"{item_info.original_size[0]}",
        "height": f"{item_info.original_size[1]}",
        "format_version": "1.0.1",
    }
    if item_info.frame_count is not None:
        metadata["frame_count"] = f"{item_info.frame_count}"

    for key, value in sd.items():
        # NaN check and show warning, replace NaN with 0
        if torch.isnan(value).any():
            logger.warning(f"{key} tensor has NaN: {item_info.item_key}, replace NaN with 0")
            value[torch.isnan(value)] = 0

    latent_dir = os.path.dirname(item_info.latent_cache_path)
    os.makedirs(latent_dir, exist_ok=True)

    save_file(sd, item_info.latent_cache_path, metadata=metadata)


def save_text_encoder_output_cache(item_info: ItemInfo, embed: torch.Tensor, mask: Optional[torch.Tensor], is_llm: bool):
    """HunyuanVideo architecture only"""
    assert (
        embed.dim() == 1 or embed.dim() == 2
    ), f"embed should be 2D tensor (feature, hidden_size) or (hidden_size,), got {embed.shape}"
    assert mask is None or mask.dim() == 1, f"mask should be 1D tensor (feature), got {mask.shape}"

    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    text_encoder_type = "llm" if is_llm else "clipL"
    sd[f"{text_encoder_type}_{dtype_str}"] = embed.detach().cpu()
    if mask is not None:
        sd[f"{text_encoder_type}_mask"] = mask.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_HUNYUAN_VIDEO_FULL)


def save_text_encoder_output_cache_wan(item_info: ItemInfo, embed: torch.Tensor):
    """Wan architecture only. Wan2.1 only has a single text encoder"""

    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    text_encoder_type = "t5"
    sd[f"varlen_{text_encoder_type}_{dtype_str}"] = embed.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_WAN_FULL)


def save_text_encoder_output_cache_framepack(
    item_info: ItemInfo, llama_vec: torch.Tensor, llama_attention_mask: torch.Tensor, clip_l_pooler: torch.Tensor
):
    """FramePack architecture only."""
    sd = {}
    dtype_str = dtype_to_str(llama_vec.dtype)
    sd[f"llama_vec_{dtype_str}"] = llama_vec.detach().cpu()
    sd[f"llama_attention_mask"] = llama_attention_mask.detach().cpu()
    dtype_str = dtype_to_str(clip_l_pooler.dtype)
    sd[f"clip_l_pooler_{dtype_str}"] = clip_l_pooler.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_FRAMEPACK_FULL)


def save_text_encoder_output_cache_common(item_info: ItemInfo, sd: dict[str, torch.Tensor], arch_fullname: str):
    for key, value in sd.items():
        # NaN check and show warning, replace NaN with 0
        if torch.isnan(value).any():
            logger.warning(f"{key} tensor has NaN: {item_info.item_key}, replace NaN with 0")
            value[torch.isnan(value)] = 0

    metadata = {
        "architecture": arch_fullname,
        "caption1": item_info.caption,
        "format_version": "1.0.1",
    }

    if os.path.exists(item_info.text_encoder_output_cache_path):
        # load existing cache and update metadata
        with safetensors_utils.MemoryEfficientSafeOpen(item_info.text_encoder_output_cache_path) as f:
            existing_metadata = f.metadata()
            for key in f.keys():
                if key not in sd:  # avoid overwriting by existing cache, we keep the new one
                    sd[key] = f.get_tensor(key)

        assert existing_metadata["architecture"] == metadata["architecture"], "architecture mismatch"
        if existing_metadata["caption1"] != metadata["caption1"]:
            logger.warning(f"caption mismatch: existing={existing_metadata['caption1']}, new={metadata['caption1']}, overwrite")
        # TODO verify format_version

        existing_metadata.pop("caption1", None)
        existing_metadata.pop("format_version", None)
        metadata.update(existing_metadata)  # copy existing metadata except caption and format_version
    else:
        text_encoder_output_dir = os.path.dirname(item_info.text_encoder_output_cache_path)
        os.makedirs(text_encoder_output_dir, exist_ok=True)

    safetensors_utils.mem_eff_save_file(sd, item_info.text_encoder_output_cache_path, metadata=metadata)


class BucketSelector:
    RESOLUTION_STEPS_HUNYUAN = 16
    RESOLUTION_STEPS_WAN = 16
    RESOLUTION_STEPS_FRAMEPACK = 16

    def __init__(
        self, resolution: Tuple[int, int], enable_bucket: bool = True, no_upscale: bool = False, architecture: str = "no_default"
    ):
        self.resolution = resolution
        self.bucket_area = resolution[0] * resolution[1]
        self.architecture = architecture

        if self.architecture == ARCHITECTURE_HUNYUAN_VIDEO:
            self.reso_steps = BucketSelector.RESOLUTION_STEPS_HUNYUAN
        elif self.architecture == ARCHITECTURE_WAN:
            self.reso_steps = BucketSelector.RESOLUTION_STEPS_WAN
        elif self.architecture == ARCHITECTURE_FRAMEPACK:
            self.reso_steps = BucketSelector.RESOLUTION_STEPS_FRAMEPACK
        else:
            raise ValueError(f"Invalid architecture: {self.architecture}")

        if not enable_bucket:
            # only define one bucket
            self.bucket_resolutions = [resolution]
            self.no_upscale = False
        else:
            # prepare bucket resolution
            self.no_upscale = no_upscale
            sqrt_size = int(math.sqrt(self.bucket_area))
            min_size = divisible_by(sqrt_size // 2, self.reso_steps)
            self.bucket_resolutions = []
            for w in range(min_size, sqrt_size + self.reso_steps, self.reso_steps):
                h = divisible_by(self.bucket_area // w, self.reso_steps)
                self.bucket_resolutions.append((w, h))
                self.bucket_resolutions.append((h, w))

            self.bucket_resolutions = list(set(self.bucket_resolutions))
            self.bucket_resolutions.sort()

        # calculate aspect ratio to find the nearest resolution
        self.aspect_ratios = np.array([w / h for w, h in self.bucket_resolutions])

    def get_bucket_resolution(self, image_size: tuple[int, int]) -> tuple[int, int]:
        """
        return the bucket resolution for the given image size, (width, height)
        """
        area = image_size[0] * image_size[1]
        if self.no_upscale and area <= self.bucket_area:
            w, h = image_size
            w = divisible_by(w, self.reso_steps)
            h = divisible_by(h, self.reso_steps)
            return w, h

        aspect_ratio = image_size[0] / image_size[1]
        ar_errors = self.aspect_ratios - aspect_ratio
        bucket_id = np.abs(ar_errors).argmin()
        return self.bucket_resolutions[bucket_id]


def load_video(
    video_path: str,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    bucket_selector: Optional[BucketSelector] = None,
    bucket_reso: Optional[tuple[int, int]] = None,
    source_fps: Optional[float] = None,
    target_fps: Optional[float] = None,
) -> list[np.ndarray]:
    """
    bucket_reso: if given, resize the video to the bucket resolution, (width, height)
    """
    if source_fps is None or target_fps is None:
        if os.path.isfile(video_path):
            container = av.open(video_path)
            video = []
            for i, frame in enumerate(container.decode(video=0)):
                if start_frame is not None and i < start_frame:
                    continue
                if end_frame is not None and i >= end_frame:
                    break
                frame = frame.to_image()

                if bucket_selector is not None and bucket_reso is None:
                    bucket_reso = bucket_selector.get_bucket_resolution(frame.size)  # calc resolution from first frame

                if bucket_reso is not None:
                    frame = resize_image_to_bucket(frame, bucket_reso)
                else:
                    frame = np.array(frame)

                video.append(frame)
            container.close()
        else:
            # load images in the directory
            image_files = glob_images(video_path)
            image_files.sort()
            video = []
            for i in range(len(image_files)):
                if start_frame is not None and i < start_frame:
                    continue
                if end_frame is not None and i >= end_frame:
                    break

                image_file = image_files[i]
                image = Image.open(image_file).convert("RGB")

                if bucket_selector is not None and bucket_reso is None:
                    bucket_reso = bucket_selector.get_bucket_resolution(image.size)  # calc resolution from first frame
                image = np.array(image)
                if bucket_reso is not None:
                    image = resize_image_to_bucket(image, bucket_reso)

                video.append(image)
    else:
        # drop frames to match the target fps TODO commonize this code with the above if this works
        frame_index_delta = target_fps / source_fps  # example: 16 / 30 = 0.5333
        if os.path.isfile(video_path):
            container = av.open(video_path)
            video = []
            frame_index_with_fraction = 0.0
            previous_frame_index = -1
            for i, frame in enumerate(container.decode(video=0)):
                target_frame_index = int(frame_index_with_fraction)
                frame_index_with_fraction += frame_index_delta

                if target_frame_index == previous_frame_index:  # drop this frame
                    continue

                # accept this frame
                previous_frame_index = target_frame_index

                if start_frame is not None and target_frame_index < start_frame:
                    continue
                if end_frame is not None and target_frame_index >= end_frame:
                    break
                frame = frame.to_image()

                if bucket_selector is not None and bucket_reso is None:
                    bucket_reso = bucket_selector.get_bucket_resolution(frame.size)  # calc resolution from first frame

                if bucket_reso is not None:
                    frame = resize_image_to_bucket(frame, bucket_reso)
                else:
                    frame = np.array(frame)

                video.append(frame)
            container.close()
        else:
            # load images in the directory
            image_files = glob_images(video_path)
            image_files.sort()
            video = []
            frame_index_with_fraction = 0.0
            previous_frame_index = -1
            for i in range(len(image_files)):
                target_frame_index = int(frame_index_with_fraction)
                frame_index_with_fraction += frame_index_delta

                if target_frame_index == previous_frame_index:  # drop this frame
                    continue

                # accept this frame
                previous_frame_index = target_frame_index

                if start_frame is not None and target_frame_index < start_frame:
                    continue
                if end_frame is not None and target_frame_index >= end_frame:
                    break

                image_file = image_files[i]
                image = Image.open(image_file).convert("RGB")

                if bucket_selector is not None and bucket_reso is None:
                    bucket_reso = bucket_selector.get_bucket_resolution(image.size)  # calc resolution from first frame
                image = np.array(image)
                if bucket_reso is not None:
                    image = resize_image_to_bucket(image, bucket_reso)

                video.append(image)

    return video


class BucketBatchManager:

    def __init__(self, bucketed_item_info: dict[tuple[Any], list[ItemInfo]], batch_size: int, control_count_per_image: int = 1):
        self.batch_size = batch_size
        self.buckets = bucketed_item_info
        self.bucket_resos = list(self.buckets.keys())
        self.bucket_resos.sort()
        self.control_count_per_image = control_count_per_image

        # indices for enumerating batches. each batch is reso + batch_idx. reso is (width, height) or (width, height, frames)
        self.bucket_batch_indices: list[tuple[tuple[Any], int]] = []
        for bucket_reso in self.bucket_resos:
            bucket = self.buckets[bucket_reso]
            num_batches = math.ceil(len(bucket) / self.batch_size)
            for i in range(num_batches):
                self.bucket_batch_indices.append((bucket_reso, i))

        # do no shuffle here to avoid multiple datasets have different order
        # self.shuffle()

    def show_bucket_info(self):
        for bucket_reso in self.bucket_resos:
            bucket = self.buckets[bucket_reso]
            logger.info(f"bucket: {bucket_reso}, count: {len(bucket)}")

        logger.info(f"total batches: {len(self)}")

    def shuffle(self):
        # shuffle each bucket
        for bucket in self.buckets.values():
            random.shuffle(bucket)

        # shuffle the order of batches
        random.shuffle(self.bucket_batch_indices)

    def __len__(self):
        return len(self.bucket_batch_indices)

    def __getitem__(self, idx):
        bucket_reso, batch_idx = self.bucket_batch_indices[idx]
        bucket = self.buckets[bucket_reso]
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, len(bucket))

        batch_tensor_data = {}
        varlen_keys = set()
        for item_info in bucket[start:end]:
            sd_latent = load_file(item_info.latent_cache_path)
            sd_te = load_file(item_info.text_encoder_output_cache_path)
            sd = {**sd_latent, **sd_te}

            # TODO refactor this
            for key in sd.keys():
                is_varlen_key = key.startswith("varlen_")  # varlen keys are not stacked
                content_key = key

                if is_varlen_key:
                    content_key = content_key.replace("varlen_", "")

                if content_key.endswith("_mask"):
                    pass
                else:
                    content_key = content_key.rsplit("_", 1)[0]  # remove dtype
                    if any([content_key.startswith(x) for x in ['latents_', 'target_latent_']]):
                        content_key = content_key.rsplit("_", 1)[0]  # remove FxHxW

                if content_key not in batch_tensor_data:
                    batch_tensor_data[content_key] = []
                batch_tensor_data[content_key].append(sd[key])

                if is_varlen_key:
                    varlen_keys.add(content_key)

        for key in batch_tensor_data.keys():
            if key not in varlen_keys:
                batch_tensor_data[key] = torch.stack(batch_tensor_data[key])
            if key.startswith("latents_clean"):
                batch_tensor_data[key] = batch_tensor_data[key][:,:,:self.control_count_per_image,:,:]
            if key.startswith("target_latent_masks"):
                batch_tensor_data[key] = batch_tensor_data[key][:,:,:self.control_count_per_image,:,:]
            if key.startswith("clean_latent_bboxes"):
                batch_tensor_data[key] = batch_tensor_data[key][:,:self.control_count_per_image,:]
                
        return batch_tensor_data


class ContentDatasource:
    def __init__(self):
        self.caption_only = False  # set to True to only fetch caption for Text Encoder caching
        self.has_control = False

    def set_caption_only(self, caption_only: bool):
        self.caption_only = caption_only

    def is_indexable(self):
        return False

    def get_caption(self, idx: int) -> tuple[str, str]:
        """
        Returns caption. May not be called if is_indexable() returns False.
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class ImageDatasource(ContentDatasource):
    def __init__(self):
        super().__init__()

    def get_image_data(self, idx: int) -> tuple[str, Image.Image, str]:
        """
        Returns image data as a tuple of image path, image, and caption for the given index.
        Key must be unique and valid as a file name.
        May not be called if is_indexable() returns False.
        """
        raise NotImplementedError


class ImageDirectoryDatasource(ImageDatasource):
    def __init__(
        self,
        image_directory: str,
        caption_extension: Optional[str] = None,
        control_directory: Optional[str] = None,
        control_count_per_image: int = 1,
    ):
        super().__init__()
        self.image_directory = image_directory
        self.caption_extension = caption_extension
        self.control_directory = control_directory
        self.control_count_per_image = control_count_per_image
        self.current_idx = 0

        # glob images
        logger.info(f"glob images in {self.image_directory}")
        self.image_paths = glob_images(self.image_directory)
        logger.info(f"found {len(self.image_paths)} images")

        # glob control images if specified
        if self.control_directory is not None:
            logger.info(f"glob control images in {self.control_directory}")
            self.has_control = True
            self.control_paths = {}
            for image_path in self.image_paths:
                image_basename = os.path.basename(image_path)
                image_basename_no_ext = os.path.splitext(image_basename)[0]
                potential_paths = glob.glob(os.path.join(self.control_directory, os.path.splitext(image_basename)[0] + "*.*"))
                if potential_paths:
                    # sort by the digits (`_0000`) suffix, prefer the one without the suffix
                    def sort_key(path):
                        basename = os.path.basename(path)
                        basename_no_ext = os.path.splitext(basename)[0]
                        if image_basename_no_ext == basename_no_ext:  # prefer the one without suffix
                            return 0
                        digits_suffix = basename_no_ext.rsplit("_", 1)[-1]
                        if not digits_suffix.isdigit():
                            raise ValueError(f"Invalid digits suffix in {basename_no_ext}")
                        return int(digits_suffix) + 1

                    potential_paths.sort(key=sort_key)
                    if len(potential_paths) < control_count_per_image:
                        logger.error(
                            f"Not enough control images for {image_path}: found {len(potential_paths)}, expected {control_count_per_image}"
                        )
                        raise ValueError(
                            f"Not enough control images for {image_path}: found {len(potential_paths)}, expected {control_count_per_image}"
                        )

                    # take the first `control_count_per_image` paths
                    self.control_paths[image_path] = potential_paths[:control_count_per_image]
            logger.info(f"found {len(self.control_paths)} matching control images")

            missing_controls = len(self.image_paths) - len(self.control_paths)
            if missing_controls > 0:
                missing_control_paths = set(self.image_paths) - set(self.control_paths.keys())
                logger.error(f"Could not find matching control images for {missing_controls} images: {missing_control_paths}")
                raise ValueError(f"Could not find matching control images for {missing_controls} images")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.image_paths)

    def get_image_data(self, idx: int) -> tuple[str, Image.Image, str, Optional[Image.Image]]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        _, caption = self.get_caption(idx)

        controls = None
        if self.has_control:
            controls = []
            for control_path in self.control_paths[image_path]:
                control = Image.open(control_path)
                if control.mode != "RGB" and control.mode != "RGBA":
                    control = control.convert("RGB")
                controls.append(control)

        return image_path, image, caption, controls, None

    def get_caption(self, idx: int) -> tuple[str, str]:
        image_path = self.image_paths[idx]
        caption_path = os.path.splitext(image_path)[0] + self.caption_extension if self.caption_extension else ""
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()
        return image_path, caption

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> callable:
        """
        Returns a fetcher function that returns image data.
        """
        if self.current_idx >= len(self.image_paths):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)
        else:

            def create_image_fetcher(index):
                return lambda: self.get_image_data(index)

            fetcher = create_image_fetcher(self.current_idx)

        self.current_idx += 1
        return fetcher


class ImageJsonlDatasource(ImageDatasource):
    def __init__(self, image_jsonl_file: str, control_count_per_image: int = 1):
        super().__init__()
        self.image_jsonl_file = image_jsonl_file
        self.control_count_per_image = control_count_per_image
        self.current_idx = 0

        # load jsonl
        logger.info(f"load image jsonl from {self.image_jsonl_file}")
        self.data = []
        with open(self.image_jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.error(f"failed to load json: {line} @ {self.image_jsonl_file}")
                    raise
                self.data.append(data)
        logger.info(f"loaded {len(self.data)} images")

        # Normalize control paths
        for item in self.data:
            if "control_path" in item:
                item["control_path_0"] = item.pop("control_path")
            if "control_mask_path" in item:
                item['control_mask_path_0'] = item.pop('control_path')

            # Ensure control paths are named consistently, from control_path_0000 to control_path_0, control_path_1, etc.
            control_path_keys = [key for key in item.keys() if key.startswith("control_path_")]
            control_path_keys.sort(key=lambda x: int(x.split("_")[-1]))
            for i, key in enumerate(control_path_keys):
                if key != f"control_path_{i}":
                    item[f"control_path_{i}"] = item.pop(key)

            control_mask_path_keys = [key for key in item.keys() if key.startswith("control_mask_path_")]
            control_mask_path_keys.sort(key=lambda x: int(x.split("_")[-1]))
            for i, key in enumerate(control_mask_path_keys):
                if key != f"control_mask_path_{i}":
                    item[f"control_mask_path_{i}"] = item.pop(key)

        # Check if there are control paths in the JSONL
        self.has_control = any("control_path_0" in item for item in self.data)
        if self.has_control:
            missing_control_images = [
                item["image_path"]
                for item in self.data
                # if sum(f"control_path_{i}" not in item for i in range(self.control_count_per_image)) > 0
                if not any(f"control_path_{i}" in item for i in range(self.control_count_per_image))
            ]
            if missing_control_images:
                logger.error(f"Some images do not have control paths in JSONL data: {missing_control_images}")
                raise ValueError(f"Some images do not have control paths in JSONL data: {missing_control_images}")
            logger.info(f"found {len(self.data)} images with {self.control_count_per_image} control images per image in JSONL data")

        self.has_control_mask = any("control_mask_path_0" in item for item in self.data)
        if self.has_control_mask:
            missing_control_mask_images = [
                item["image_path"]
                for item in self.data
                # if sum(f"control_mask_path_{i}" not in item for i in range(self.control_count_per_image)) > 0
                if not any(f"control_mask_path_{i}" in item for i in range(self.control_count_per_image))
            ]
            if missing_control_mask_images:
                logger.error(f"Some images do not have control mask paths in JSONL data: {missing_control_mask_images}")
                raise ValueError(f"Some images do not have control mask paths in JSONL data: {missing_control_mask_images}")
            logger.info(f"found {len(self.data)} masks with {self.control_count_per_image} control masks per image in JSONL data")
        
        self.has_embed = any("embed_path" in item for item in self.data)
        if self.has_embed:
            missing_embed_images = [
                item["image_path"]
                for item in self.data
                if "embed_path" not in item
            ]
            if missing_embed_images:
                logger.error(f"Some images do not have embed paths in JSONL data: {missing_embed_images}")
                raise ValueError(f"Some images do not have embed paths in JSONL data: {missing_embed_images}")  

        self.has_clean_bbox = any("meta" in item for item in self.data)
        if self.has_clean_bbox:
            missing_bbox_images = [
                item["image_path"]
                for item in self.data
                if "meta" not in item
            ]
            if missing_bbox_images:
                logger.error(f"Some images do not have clean bbox paths in JSONL data: {missing_bbox_images}")
                raise ValueError(f"Some images do not have clean bbox paths in JSONL data: {missing_bbox_images}")
            logger.info(f"found {len(self.data)} metadata with {self.control_count_per_image} bbox paths per image in JSONL data")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.data)

    def get_image_data(self, idx: int) -> tuple[str, Image.Image, str, Optional[list[Image.Image]]]:
        data = self.data[idx]
        image_path = data["image_path"]
        image = Image.open(image_path).convert("RGB")
        caption = data["caption"]

        controls = None
        if self.has_control:
            controls = []
            for i in range(self.control_count_per_image):
                if f"control_path_{i}" in data:
                    control = Image.open(data[f"control_path_{i}"]).convert("RGB")
                    if self.has_control_mask and f"control_mask_path_{i}" in data:
                        control_mask = Image.open(data[f"control_mask_path_{i}"]).convert("L")
                        control = (control, control_mask)
                    controls.append(control)

        embed = None
        if self.has_embed:
            embed = Image.open(data['embed_path']).convert("RGB")

        bboxes = None
        if self.has_clean_bbox:
            bboxes = get_bbox_from_meta(data['meta'], self.control_count_per_image)
            # clean_latent_bbox = [meta['target_body'].get(str(x), [0.0,0.0,meta['width'],meta['height']]) for x in keys]
            # clean_latent_bbox = [[x[0]/meta['width'], x[1]/meta['height'], x[2]/meta['width'], x[3]/meta['height']] for x in clean_latent_bbox]
        return image_path, image, caption, controls, embed, bboxes

    def get_caption(self, idx: int) -> tuple[str, str]:
        data = self.data[idx]
        image_path = data["image_path"]
        caption = data["caption"]
        return image_path, caption

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> callable:
        if self.current_idx >= len(self.data):
            raise StopIteration

        if self.caption_only:
            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)
            fetcher = create_caption_fetcher(self.current_idx)
        else:
            def create_fetcher(index):
                return lambda: self.get_image_data(index)
            fetcher = create_fetcher(self.current_idx)

        self.current_idx += 1
        return fetcher


class VideoDatasource(ContentDatasource):
    def __init__(self):
        super().__init__()

        # None means all frames
        self.start_frame = None
        self.end_frame = None

        self.bucket_selector = None

        self.source_fps = None
        self.target_fps = None

    def __len__(self):
        raise NotImplementedError

    def get_video_data_from_path(
        self,
        video_path: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        bucket_selector: Optional[BucketSelector] = None,
    ) -> tuple[str, list[Image.Image], str]:
        # this method can resize the video if bucket_selector is given to reduce the memory usage

        start_frame = start_frame if start_frame is not None else self.start_frame
        end_frame = end_frame if end_frame is not None else self.end_frame
        bucket_selector = bucket_selector if bucket_selector is not None else self.bucket_selector

        video = load_video(
            video_path, start_frame, end_frame, bucket_selector, source_fps=self.source_fps, target_fps=self.target_fps
        )
        return video

    def get_control_data_from_path(
        self,
        control_path: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        bucket_selector: Optional[BucketSelector] = None,
    ) -> list[Image.Image]:
        start_frame = start_frame if start_frame is not None else self.start_frame
        end_frame = end_frame if end_frame is not None else self.end_frame
        bucket_selector = bucket_selector if bucket_selector is not None else self.bucket_selector

        control = load_video(
            control_path, start_frame, end_frame, bucket_selector, source_fps=self.source_fps, target_fps=self.target_fps
        )
        return control

    def set_start_and_end_frame(self, start_frame: Optional[int], end_frame: Optional[int]):
        self.start_frame = start_frame
        self.end_frame = end_frame

    def set_bucket_selector(self, bucket_selector: BucketSelector):
        self.bucket_selector = bucket_selector

    def set_source_and_target_fps(self, source_fps: Optional[float], target_fps: Optional[float]):
        self.source_fps = source_fps
        self.target_fps = target_fps

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class VideoDirectoryDatasource(VideoDatasource):
    def __init__(self, video_directory: str, caption_extension: Optional[str] = None, control_directory: Optional[str] = None):
        super().__init__()
        self.video_directory = video_directory
        self.caption_extension = caption_extension
        self.control_directory = control_directory  # 新しく追加: コントロール画像ディレクトリ
        self.current_idx = 0

        # glob videos
        logger.info(f"glob videos in {self.video_directory}")
        self.video_paths = glob_videos(self.video_directory)
        logger.info(f"found {len(self.video_paths)} videos")

        # glob control images if specified
        if self.control_directory is not None:
            logger.info(f"glob control videos in {self.control_directory}")
            self.has_control = True
            self.control_paths = {}
            for video_path in self.video_paths:
                video_basename = os.path.basename(video_path)
                # construct control path from video path
                # for example: video_path = "vid/video.mp4" -> control_path = "control/video.mp4"
                control_path = os.path.join(self.control_directory, video_basename)
                if os.path.exists(control_path):
                    self.control_paths[video_path] = control_path
                else:
                    # use the same base name for control path
                    base_name = os.path.splitext(video_basename)[0]

                    # directory with images. for example: video_path = "vid/video.mp4" -> control_path = "control/video"
                    potential_path = os.path.join(self.control_directory, base_name)  # no extension
                    if os.path.isdir(potential_path):
                        self.control_paths[video_path] = potential_path
                    else:
                        # another extension for control path
                        # for example: video_path = "vid/video.mp4" -> control_path = "control/video.mov"
                        for ext in VIDEO_EXTENSIONS:
                            potential_path = os.path.join(self.control_directory, base_name + ext)
                            if os.path.exists(potential_path):
                                self.control_paths[video_path] = potential_path
                                break

            logger.info(f"found {len(self.control_paths)} matching control videos/images")
            # check if all videos have matching control paths, if not, raise an error
            missing_controls = len(self.video_paths) - len(self.control_paths)
            if missing_controls > 0:
                # logger.warning(f"Could not find matching control videos/images for {missing_controls} videos")
                missing_controls_videos = [video_path for video_path in self.video_paths if video_path not in self.control_paths]
                logger.error(
                    f"Could not find matching control videos/images for {missing_controls} videos: {missing_controls_videos}"
                )
                raise ValueError(f"Could not find matching control videos/images for {missing_controls} videos")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.video_paths)

    def get_video_data(
        self,
        idx: int,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        bucket_selector: Optional[BucketSelector] = None,
    ) -> tuple[str, list[Image.Image], str, Optional[list[Image.Image]]]:
        video_path = self.video_paths[idx]
        video = self.get_video_data_from_path(video_path, start_frame, end_frame, bucket_selector)

        _, caption = self.get_caption(idx)

        control = None
        if self.control_directory is not None and video_path in self.control_paths:
            control_path = self.control_paths[video_path]
            control = self.get_control_data_from_path(control_path, start_frame, end_frame, bucket_selector)

        return video_path, video, caption, control

    def get_caption(self, idx: int) -> tuple[str, str]:
        video_path = self.video_paths[idx]
        caption_path = os.path.splitext(video_path)[0] + self.caption_extension if self.caption_extension else ""
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()
        return video_path, caption

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.video_paths):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)

        else:

            def create_fetcher(index):
                return lambda: self.get_video_data(index)

            fetcher = create_fetcher(self.current_idx)

        self.current_idx += 1
        return fetcher


class VideoJsonlDatasource(VideoDatasource):
    def __init__(self, video_jsonl_file: str):
        super().__init__()
        self.video_jsonl_file = video_jsonl_file
        self.current_idx = 0

        # load jsonl
        logger.info(f"load video jsonl from {self.video_jsonl_file}")
        self.data = []
        with open(self.video_jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.data.append(data)
        logger.info(f"loaded {len(self.data)} videos")

        # Check if there are control paths in the JSONL
        self.has_control = any("control_path" in item for item in self.data)
        if self.has_control:
            control_count = sum(1 for item in self.data if "control_path" in item)
            if control_count < len(self.data):
                missing_control_videos = [item["video_path"] for item in self.data if "control_path" not in item]
                logger.error(f"Some videos do not have control paths in JSONL data: {missing_control_videos}")
                raise ValueError(f"Some videos do not have control paths in JSONL data: {missing_control_videos}")
            logger.info(f"found {control_count} control videos/images in JSONL data")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.data)

    def get_video_data(
        self,
        idx: int,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        bucket_selector: Optional[BucketSelector] = None,
    ) -> tuple[str, list[Image.Image], str, Optional[list[Image.Image]]]:
        data = self.data[idx]
        video_path = data["video_path"]
        video = self.get_video_data_from_path(video_path, start_frame, end_frame, bucket_selector)

        caption = data["caption"]

        control = None
        if "control_path" in data and data["control_path"]:
            control_path = data["control_path"]
            control = self.get_control_data_from_path(control_path, start_frame, end_frame, bucket_selector)

        return video_path, video, caption, control

    def get_caption(self, idx: int) -> tuple[str, str]:
        data = self.data[idx]
        video_path = data["video_path"]
        caption = data["caption"]
        return video_path, caption

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.data):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)

        else:

            def create_fetcher(index):
                return lambda: self.get_video_data(index)

            fetcher = create_fetcher(self.current_idx)

        self.current_idx += 1
        return fetcher


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        resolution: Tuple[int, int] = (960, 544),
        caption_extension: Optional[str] = None,
        batch_size: int = 1,
        num_repeats: int = 1,
        enable_bucket: bool = False,
        bucket_no_upscale: bool = False,
        cache_directory: Optional[str] = None,
        debug_dataset: bool = False,
        architecture: str = "no_default",
    ):
        self.resolution = resolution
        self.caption_extension = caption_extension
        self.batch_size = batch_size
        self.num_repeats = num_repeats
        self.enable_bucket = enable_bucket
        self.bucket_no_upscale = bucket_no_upscale
        self.cache_directory = cache_directory
        self.debug_dataset = debug_dataset
        self.architecture = architecture
        self.seed = None
        self.current_epoch = 0

        if not self.enable_bucket:
            self.bucket_no_upscale = False

    def get_metadata(self) -> dict:
        metadata = {
            "resolution": self.resolution,
            "caption_extension": self.caption_extension,
            "batch_size_per_device": self.batch_size,
            "num_repeats": self.num_repeats,
            "enable_bucket": bool(self.enable_bucket),
            "bucket_no_upscale": bool(self.bucket_no_upscale),
        }
        return metadata

    def get_all_latent_cache_files(self):
        return glob.glob(os.path.join(self.cache_directory, f"*_{self.architecture}.safetensors"))

    def get_all_text_encoder_output_cache_files(self):
        return glob.glob(os.path.join(self.cache_directory, f"*_{self.architecture}_te.safetensors"))

    def get_latent_cache_path(self, item_info: ItemInfo) -> str:
        """
        Returns the cache path for the latent tensor.

        item_info: ItemInfo object

        Returns:
            str: cache path

        cache_path is based on the item_key and the resolution.
        """
        w, h = item_info.original_size
        basename = os.path.splitext(os.path.basename(item_info.item_key))[0]
        # basename = os.path.basename(os.path.dirname(item_info.item_key))
        assert self.cache_directory is not None, "cache_directory is required / cache_directoryは必須です"
        return os.path.join(self.cache_directory, f"{basename}_{w:04d}x{h:04d}_{self.architecture}.safetensors")

    def get_text_encoder_output_cache_path(self, item_info: ItemInfo) -> str:
        basename = os.path.splitext(os.path.basename(item_info.item_key))[0]
        # basename = os.path.basename(os.path.dirname(item_info.item_key))
        assert self.cache_directory is not None, "cache_directory is required / cache_directoryは必須です"
        return os.path.join(self.cache_directory, f"{basename}_{self.architecture}_te.safetensors")

    def retrieve_latent_cache_batches(self, num_workers: int):
        raise NotImplementedError

    def retrieve_text_encoder_output_cache_batches(self, num_workers: int):
        raise NotImplementedError

    def prepare_for_training(self):
        pass

    def set_seed(self, seed: int):
        self.seed = seed

    def set_current_epoch(self, epoch):
        if not self.current_epoch == epoch:  # shuffle buckets when epoch is incremented
            if epoch > self.current_epoch:
                logger.info("epoch is incremented. current_epoch: {}, epoch: {}".format(self.current_epoch, epoch))
                num_epochs = epoch - self.current_epoch
                for _ in range(num_epochs):
                    self.current_epoch += 1
                    self.shuffle_buckets()
                # self.current_epoch seem to be set to 0 again in the next epoch. it may be caused by skipped_dataloader?
            else:
                logger.warning("epoch is not incremented. current_epoch: {}, epoch: {}".format(self.current_epoch, epoch))
                self.current_epoch = epoch

    def set_current_step(self, step):
        self.current_step = step

    def set_max_train_steps(self, max_train_steps):
        self.max_train_steps = max_train_steps

    def shuffle_buckets(self):
        raise NotImplementedError

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def _default_retrieve_text_encoder_output_cache_batches(self, datasource: ContentDatasource, batch_size: int, num_workers: int):
        datasource.set_caption_only(True)
        executor = ThreadPoolExecutor(max_workers=num_workers)

        data: list[ItemInfo] = []
        futures = []

        def aggregate_future(consume_all: bool = False):
            while len(futures) >= num_workers or (consume_all and len(futures) > 0):
                completed_futures = [future for future in futures if future.done()]
                if len(completed_futures) == 0:
                    if len(futures) >= num_workers or consume_all:  # to avoid adding too many futures
                        time.sleep(0.1)
                        continue
                    else:
                        break  # submit batch if possible

                for future in completed_futures:
                    item_key, caption = future.result()
                    item_info = ItemInfo(item_key, caption, (0, 0), (0, 0))
                    item_info.text_encoder_output_cache_path = self.get_text_encoder_output_cache_path(item_info)
                    data.append(item_info)

                    futures.remove(future)

        def submit_batch(flush: bool = False):
            nonlocal data
            if len(data) >= batch_size or (len(data) > 0 and flush):
                batch = data[0:batch_size]
                if len(data) > batch_size:
                    data = data[batch_size:]
                else:
                    data = []
                return batch
            return None

        for fetch_op in datasource:
            future = executor.submit(fetch_op)
            futures.append(future)
            aggregate_future()
            while True:
                batch = submit_batch()
                if batch is None:
                    break
                yield batch

        aggregate_future(consume_all=True)
        while True:
            batch = submit_batch(flush=True)
            if batch is None:
                break
            yield batch

        executor.shutdown()


class ImageDataset(BaseDataset):
    def __init__(
        self,
        resolution: Tuple[int, int],
        caption_extension: Optional[str],
        batch_size: int,
        num_repeats: int,
        enable_bucket: bool,
        bucket_no_upscale: bool,
        image_directory: Optional[str] = None,
        image_jsonl_file: Optional[str] = None,
        control_directory: Optional[str] = None,
        cache_directory: Optional[str] = None,
        fp_latent_window_size: Optional[int] = 9,
        fp_1f_clean_indices: Optional[list[int]] = None,
        fp_1f_target_index: Optional[int] = None,
        fp_1f_no_post: Optional[bool] = False,
        debug_dataset: bool = False,
        architecture: str = "no_default",
        control_resolution: Tuple[int, int] = None,
        control_count_per_image: int = 1,
    ):
        super(ImageDataset, self).__init__(
            resolution,
            caption_extension,
            batch_size,
            num_repeats,
            enable_bucket,
            bucket_no_upscale,
            cache_directory,
            debug_dataset,
            architecture,
        )
        self.image_directory = image_directory
        self.image_jsonl_file = image_jsonl_file
        self.control_directory = control_directory
        self.fp_latent_window_size = fp_latent_window_size
        self.fp_1f_clean_indices = fp_1f_clean_indices
        self.fp_1f_target_index = fp_1f_target_index
        self.fp_1f_no_post = fp_1f_no_post
        self.control_count_per_image = control_count_per_image

        # if fp_1f_clean_indices is not None:
        #     self.control_count_per_image = len(fp_1f_clean_indices)

        if image_directory is not None:
            self.datasource = ImageDirectoryDatasource(
                image_directory, caption_extension, control_directory, self.control_count_per_image
            )
        elif image_jsonl_file is not None:
            self.datasource = ImageJsonlDatasource(image_jsonl_file, self.control_count_per_image)
        else:
            raise ValueError("image_directory or image_jsonl_file must be specified")

        if self.cache_directory is None:
            self.cache_directory = self.image_directory

        self.batch_manager = None
        self.num_train_items = 0
        self.has_control = self.datasource.has_control
        self.control_resolution = control_resolution

    def get_metadata(self):
        metadata = super().get_metadata()
        if self.image_directory is not None:
            metadata["image_directory"] = os.path.basename(self.image_directory)
        if self.image_jsonl_file is not None:
            metadata["image_jsonl_file"] = os.path.basename(self.image_jsonl_file)
        if self.control_directory is not None:
            metadata["control_directory"] = os.path.basename(self.control_directory)
        metadata["has_control"] = self.has_control
        return metadata

    def get_total_image_count(self):
        return len(self.datasource) if self.datasource.is_indexable() else None

    def retrieve_latent_cache_batches(self, num_workers: int):
        buckset_selector = BucketSelector(self.resolution, self.enable_bucket, self.bucket_no_upscale, self.architecture)
        executor = ThreadPoolExecutor(max_workers=num_workers)

        batches: dict[tuple[int, int], list[ItemInfo]] = {}  # (width, height) -> [ItemInfo]
        futures = []

        # aggregate futures and sort by bucket resolution
        def aggregate_future(consume_all: bool = False):
            while len(futures) >= num_workers or (consume_all and len(futures) > 0):
                completed_futures = [future for future in futures if future.done()]
                if len(completed_futures) == 0:
                    if len(futures) >= num_workers or consume_all:  # to avoid adding too many futures
                        time.sleep(0.1)
                        continue
                    else:
                        break  # submit batch if possible

                for future in completed_futures:
                    original_size, item_key, image, caption, controls, embed, clean_latent_bbox = future.result()
                    bucket_height, bucket_width = image.shape[:2]
                    bucket_reso = (bucket_width, bucket_height)

                    item_info = ItemInfo(item_key, caption, original_size, bucket_reso, content=image)
                    item_info.latent_cache_path = self.get_latent_cache_path(item_info)
                    item_info.fp_latent_window_size = self.fp_latent_window_size
                    item_info.fp_1f_clean_indices = self.fp_1f_clean_indices
                    item_info.fp_1f_target_index = self.fp_1f_target_index
                    item_info.fp_1f_no_post = self.fp_1f_no_post

                    if self.architecture == ARCHITECTURE_FRAMEPACK or self.architecture == ARCHITECTURE_WAN:
                        # we need to split the bucket with latent window size and optional 1f clean indices, zero post
                        bucket_reso = list(bucket_reso) + [self.fp_latent_window_size]
                        if self.fp_1f_clean_indices is not None:
                            bucket_reso.append(len(self.fp_1f_clean_indices))
                            bucket_reso.append(self.fp_1f_no_post)
                        bucket_reso = tuple(bucket_reso)

                    if controls is not None:
                        item_info.control_content = controls
                    if embed is not None:
                        item_info.embed_content = embed
                    if clean_latent_bbox is not None:
                        item_info.clean_latent_bboxes = clean_latent_bbox

                    if bucket_reso not in batches:
                        batches[bucket_reso] = []
                    batches[bucket_reso].append(item_info)

                    futures.remove(future)

        # submit batch if some bucket has enough items
        def submit_batch(flush: bool = False):
            for key in batches:
                if len(batches[key]) >= self.batch_size or flush:
                    batch = batches[key][0 : self.batch_size]
                    if len(batches[key]) > self.batch_size:
                        batches[key] = batches[key][self.batch_size :]
                    else:
                        del batches[key]
                    return key, batch
            return None, None

        for fetch_op in self.datasource:

            # fetch and resize image in a separate thread
            def fetch_and_resize(op: callable) -> tuple[tuple[int, int], str, Image.Image, str, Optional[Image.Image]]:
                image_key, image, caption, controls, embed, bboxes = op()
                image: Image.Image
                image_size = image.size

                bucket_reso = buckset_selector.get_bucket_resolution(image_size)
                image = resize_image_to_bucket(image, bucket_reso)  # returns np.ndarray

                resized_controls = None
                if controls is not None:
                    control_bucket_reso = self.control_resolution
                    if control_bucket_reso is None:
                        control_bucket_reso = bucket_reso
                    resized_controls = []
                    for control in controls:
                        if type(control) == tuple:
                            control, control_mask = control
                            resized_control = resize_image_to_bucket(control, control_bucket_reso) 
                            resized_control_mask = resize_image_to_bucket(control_mask, bucket_reso) 
                            resized_control = (resized_control, resized_control_mask)
                        else:
                            resized_control = resize_image_to_bucket(control, control_bucket_reso)  # returns np.ndarray
                        resized_controls.append(resized_control)

                resized_embed = None
                if embed is not None:
                    bucket_reso = buckset_selector.get_bucket_resolution(embed.size)
                    resized_embed = resize_image_to_bucket(embed, bucket_reso)

                face_bboxes = []
                if bboxes is not None:
                    for bbox in bboxes:
                        face_bbox = get_facebbox_from_bbox(bbox, 
                            self.control_resolution[0],
                            self.control_resolution[1],
                            bucket_reso[0], bucket_reso[1],
                            face_bbox = None,
                            mode = "provided_size_mid_x"
                        )
                        # control_bucket_reso = self.control_resolution
                        # face_bbox = np.array([bbox[0], bbox[1], 
                        #     bbox[0]+(control_bucket_reso[1] / bucket_reso[1]), 
                        #     bbox[1]+(control_bucket_reso[0] / bucket_reso[0])])
                        face_bboxes.append(face_bbox)
                face_bboxes = torch.tensor(face_bboxes).float()
                return image_size, image_key, image, caption, resized_controls, resized_embed, face_bboxes

            future = executor.submit(fetch_and_resize, fetch_op)
            futures.append(future)
            aggregate_future()
            while True:
                key, batch = submit_batch()
                if key is None:
                    break
                yield key, batch

        aggregate_future(consume_all=True)
        while True:
            key, batch = submit_batch(flush=True)
            if key is None:
                break
            yield key, batch

        executor.shutdown()

    def retrieve_text_encoder_output_cache_batches(self, num_workers: int):
        return self._default_retrieve_text_encoder_output_cache_batches(self.datasource, self.batch_size, num_workers)

    def prepare_for_training(self):
        bucket_selector = BucketSelector(self.resolution, self.enable_bucket, self.bucket_no_upscale, self.architecture)

        # glob cache files
        latent_cache_files = glob.glob(os.path.join(self.cache_directory, f"*_{self.architecture}.safetensors"))

        # assign cache files to item info
        bucketed_item_info: dict[tuple[int, int], list[ItemInfo]] = {}  # (width, height) -> [ItemInfo]
        for cache_file in latent_cache_files:
            tokens = os.path.basename(cache_file).split("_")

            image_size = tokens[-2]  # 0000x0000
            image_width, image_height = map(int, image_size.split("x"))
            image_size = (image_width, image_height)

            item_key = "_".join(tokens[:-2])
            text_encoder_output_cache_file = os.path.join(self.cache_directory, f"{item_key}_{self.architecture}_te.safetensors")
            if not os.path.exists(text_encoder_output_cache_file):
                logger.warning(f"Text encoder output cache file not found: {text_encoder_output_cache_file}")
                continue

            bucket_reso = bucket_selector.get_bucket_resolution(image_size)

            if self.architecture == ARCHITECTURE_FRAMEPACK or self.architecture == ARCHITECTURE_WAN:
                # we need to split the bucket with latent window size and optional 1f clean indices, zero post
                bucket_reso = list(bucket_reso) + [self.fp_latent_window_size]
                if self.fp_1f_clean_indices is not None:
                    bucket_reso.append(len(self.fp_1f_clean_indices))
                    bucket_reso.append(self.fp_1f_no_post)
                bucket_reso = tuple(bucket_reso)

            item_info = ItemInfo(item_key, "", image_size, bucket_reso, latent_cache_path=cache_file)
            item_info.text_encoder_output_cache_path = text_encoder_output_cache_file

            bucket = bucketed_item_info.get(bucket_reso, [])
            for _ in range(self.num_repeats):
                bucket.append(item_info)
            bucketed_item_info[bucket_reso] = bucket

        # prepare batch manager
        self.batch_manager = BucketBatchManager(bucketed_item_info, self.batch_size, self.control_count_per_image)
        self.batch_manager.show_bucket_info()

        self.num_train_items = sum([len(bucket) for bucket in bucketed_item_info.values()])

    def shuffle_buckets(self):
        # set random seed for this epoch
        random.seed(self.seed + self.current_epoch)
        self.batch_manager.shuffle()

    def __len__(self):
        if self.batch_manager is None:
            return 100  # dummy value
        return len(self.batch_manager)

    def __getitem__(self, idx):
        return self.batch_manager[idx]


class VideoDataset(BaseDataset):
    TARGET_FPS_HUNYUAN = 24.0
    TARGET_FPS_WAN = 16.0
    TARGET_FPS_FRAMEPACK = 30.0

    def __init__(
        self,
        resolution: Tuple[int, int],
        caption_extension: Optional[str],
        batch_size: int,
        num_repeats: int,
        enable_bucket: bool,
        bucket_no_upscale: bool,
        frame_extraction: Optional[str] = "head",
        frame_stride: Optional[int] = 1,
        frame_sample: Optional[int] = 1,
        target_frames: Optional[list[int]] = None,
        max_frames: Optional[int] = None,
        source_fps: Optional[float] = None,
        video_directory: Optional[str] = None,
        video_jsonl_file: Optional[str] = None,
        control_directory: Optional[str] = None,
        cache_directory: Optional[str] = None,
        fp_latent_window_size: Optional[int] = 9,
        debug_dataset: bool = False,
        architecture: str = "no_default",
    ):
        super(VideoDataset, self).__init__(
            resolution,
            caption_extension,
            batch_size,
            num_repeats,
            enable_bucket,
            bucket_no_upscale,
            cache_directory,
            debug_dataset,
            architecture,
        )
        self.video_directory = video_directory
        self.video_jsonl_file = video_jsonl_file
        self.control_directory = control_directory
        self.frame_extraction = frame_extraction
        self.frame_stride = frame_stride
        self.frame_sample = frame_sample
        self.max_frames = max_frames
        self.source_fps = source_fps
        self.fp_latent_window_size = fp_latent_window_size

        if self.architecture == ARCHITECTURE_HUNYUAN_VIDEO:
            self.target_fps = VideoDataset.TARGET_FPS_HUNYUAN
        elif self.architecture == ARCHITECTURE_WAN:
            self.target_fps = VideoDataset.TARGET_FPS_WAN
        elif self.architecture == ARCHITECTURE_FRAMEPACK:
            self.target_fps = VideoDataset.TARGET_FPS_FRAMEPACK
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

        if target_frames is not None:
            target_frames = list(set(target_frames))
            target_frames.sort()

            # round each value to N*4+1
            rounded_target_frames = [(f - 1) // 4 * 4 + 1 for f in target_frames]
            rouneded_target_frames = list(set(rounded_target_frames))
            rouneded_target_frames.sort()

            # if value is changed, warn
            if target_frames != rounded_target_frames:
                logger.warning(f"target_frames are rounded to {rounded_target_frames}")

            target_frames = tuple(rounded_target_frames)

        self.target_frames = target_frames

        if video_directory is not None:
            self.datasource = VideoDirectoryDatasource(video_directory, caption_extension, control_directory)
        elif video_jsonl_file is not None:
            self.datasource = VideoJsonlDatasource(video_jsonl_file)

        if self.frame_extraction == "uniform" and self.frame_sample == 1:
            self.frame_extraction = "head"
            logger.warning("frame_sample is set to 1 for frame_extraction=uniform. frame_extraction is changed to head.")
        if self.frame_extraction == "head":
            # head extraction. we can limit the number of frames to be extracted
            self.datasource.set_start_and_end_frame(0, max(self.target_frames))

        if self.cache_directory is None:
            self.cache_directory = self.video_directory

        self.batch_manager = None
        self.num_train_items = 0
        self.has_control = self.datasource.has_control

    def get_metadata(self):
        metadata = super().get_metadata()
        if self.video_directory is not None:
            metadata["video_directory"] = os.path.basename(self.video_directory)
        if self.video_jsonl_file is not None:
            metadata["video_jsonl_file"] = os.path.basename(self.video_jsonl_file)
        if self.control_directory is not None:
            metadata["control_directory"] = os.path.basename(self.control_directory)
        metadata["frame_extraction"] = self.frame_extraction
        metadata["frame_stride"] = self.frame_stride
        metadata["frame_sample"] = self.frame_sample
        metadata["target_frames"] = self.target_frames
        metadata["max_frames"] = self.max_frames
        metadata["source_fps"] = self.source_fps
        metadata["has_control"] = self.has_control
        return metadata

    def retrieve_latent_cache_batches(self, num_workers: int):
        buckset_selector = BucketSelector(self.resolution, architecture=self.architecture)
        self.datasource.set_bucket_selector(buckset_selector)
        if self.source_fps is not None:
            self.datasource.set_source_and_target_fps(self.source_fps, self.target_fps)
        else:
            self.datasource.set_source_and_target_fps(None, None)  # no conversion

        executor = ThreadPoolExecutor(max_workers=num_workers)

        # key: (width, height, frame_count) and optional latent_window_size, value: [ItemInfo]
        batches: dict[tuple[Any], list[ItemInfo]] = {}
        futures = []

        def aggregate_future(consume_all: bool = False):
            while len(futures) >= num_workers or (consume_all and len(futures) > 0):
                completed_futures = [future for future in futures if future.done()]
                if len(completed_futures) == 0:
                    if len(futures) >= num_workers or consume_all:  # to avoid adding too many futures
                        time.sleep(0.1)
                        continue
                    else:
                        break  # submit batch if possible

                for future in completed_futures:
                    original_frame_size, video_key, video, caption, control = future.result()

                    frame_count = len(video)
                    video = np.stack(video, axis=0)
                    height, width = video.shape[1:3]
                    bucket_reso = (width, height)  # already resized

                    # process control images if available
                    control_video = None
                    if control is not None:
                        # set frame count to the same as video
                        if len(control) > frame_count:
                            control = control[:frame_count]
                        elif len(control) < frame_count:
                            # if control is shorter than video, repeat the last frame
                            last_frame = control[-1]
                            control.extend([last_frame] * (frame_count - len(control)))
                        control_video = np.stack(control, axis=0)

                    crop_pos_and_frames = []
                    if self.frame_extraction == "head":
                        for target_frame in self.target_frames:
                            if frame_count >= target_frame:
                                crop_pos_and_frames.append((0, target_frame))
                    elif self.frame_extraction == "chunk":
                        # split by target_frames
                        for target_frame in self.target_frames:
                            for i in range(0, frame_count, target_frame):
                                if i + target_frame <= frame_count:
                                    crop_pos_and_frames.append((i, target_frame))
                    elif self.frame_extraction == "slide":
                        # slide window
                        for target_frame in self.target_frames:
                            if frame_count >= target_frame:
                                for i in range(0, frame_count - target_frame + 1, self.frame_stride):
                                    crop_pos_and_frames.append((i, target_frame))
                    elif self.frame_extraction == "uniform":
                        # select N frames uniformly
                        for target_frame in self.target_frames:
                            if frame_count >= target_frame:
                                frame_indices = np.linspace(0, frame_count - target_frame, self.frame_sample, dtype=int)
                                for i in frame_indices:
                                    crop_pos_and_frames.append((i, target_frame))
                    elif self.frame_extraction == "full":
                        # select all frames
                        target_frame = min(frame_count, self.max_frames)
                        target_frame = (target_frame - 1) // 4 * 4 + 1  # round to N*4+1
                        crop_pos_and_frames.append((0, target_frame))
                    else:
                        raise ValueError(f"frame_extraction {self.frame_extraction} is not supported")

                    for crop_pos, target_frame in crop_pos_and_frames:
                        cropped_video = video[crop_pos : crop_pos + target_frame]
                        body, ext = os.path.splitext(video_key)
                        item_key = f"{body}_{crop_pos:05d}-{target_frame:03d}{ext}"
                        batch_key = (*bucket_reso, target_frame)  # bucket_reso with frame_count

                        if self.architecture == ARCHITECTURE_FRAMEPACK:
                            # add latent window size to bucket resolution
                            batch_key = (*batch_key, self.fp_latent_window_size)

                        # crop control video if available
                        cropped_control = None
                        if control_video is not None:
                            cropped_control = control_video[crop_pos : crop_pos + target_frame]

                        item_info = ItemInfo(
                            item_key, caption, original_frame_size, batch_key, frame_count=target_frame, content=cropped_video
                        )
                        item_info.latent_cache_path = self.get_latent_cache_path(item_info)
                        item_info.control_content = cropped_control  # None is allowed
                        item_info.fp_latent_window_size = self.fp_latent_window_size

                        batch = batches.get(batch_key, [])
                        batch.append(item_info)
                        batches[batch_key] = batch

                    futures.remove(future)

        def submit_batch(flush: bool = False):
            for key in batches:
                if len(batches[key]) >= self.batch_size or flush:
                    batch = batches[key][0 : self.batch_size]
                    if len(batches[key]) > self.batch_size:
                        batches[key] = batches[key][self.batch_size :]
                    else:
                        del batches[key]
                    return key, batch
            return None, None

        for operator in self.datasource:

            def fetch_and_resize(op: callable) -> tuple[tuple[int, int], str, list[np.ndarray], str, Optional[list[np.ndarray]]]:
                result = op()

                if len(result) == 3:  # for backward compatibility TODO remove this in the future
                    video_key, video, caption = result
                    control = None
                else:
                    video_key, video, caption, control = result

                video: list[np.ndarray]
                frame_size = (video[0].shape[1], video[0].shape[0])

                # resize if necessary
                bucket_reso = buckset_selector.get_bucket_resolution(frame_size)
                video = [resize_image_to_bucket(frame, bucket_reso) for frame in video]

                # resize control if necessary
                if control is not None:
                    control = [resize_image_to_bucket(frame, bucket_reso) for frame in control]

                return frame_size, video_key, video, caption, control

            future = executor.submit(fetch_and_resize, operator)
            futures.append(future)
            aggregate_future()
            while True:
                key, batch = submit_batch()
                if key is None:
                    break
                yield key, batch

        aggregate_future(consume_all=True)
        while True:
            key, batch = submit_batch(flush=True)
            if key is None:
                break
            yield key, batch

        executor.shutdown()

    def retrieve_text_encoder_output_cache_batches(self, num_workers: int):
        return self._default_retrieve_text_encoder_output_cache_batches(self.datasource, self.batch_size, num_workers)

    def prepare_for_training(self):
        bucket_selector = BucketSelector(self.resolution, self.enable_bucket, self.bucket_no_upscale, self.architecture)

        # glob cache files
        latent_cache_files = glob.glob(os.path.join(self.cache_directory, f"*_{self.architecture}.safetensors"))

        # assign cache files to item info
        bucketed_item_info: dict[tuple[int, int, int], list[ItemInfo]] = {}  # (width, height, frame_count) -> [ItemInfo]
        for cache_file in latent_cache_files:
            tokens = os.path.basename(cache_file).split("_")

            image_size = tokens[-2]  # 0000x0000
            image_width, image_height = map(int, image_size.split("x"))
            image_size = (image_width, image_height)

            frame_pos, frame_count = tokens[-3].split("-")[:2]  # "00000-000", or optional section index "00000-000-00"
            frame_pos, frame_count = int(frame_pos), int(frame_count)

            item_key = "_".join(tokens[:-3])
            text_encoder_output_cache_file = os.path.join(self.cache_directory, f"{item_key}_{self.architecture}_te.safetensors")
            if not os.path.exists(text_encoder_output_cache_file):
                logger.warning(f"Text encoder output cache file not found: {text_encoder_output_cache_file}")
                continue

            bucket_reso = bucket_selector.get_bucket_resolution(image_size)
            bucket_reso = (*bucket_reso, frame_count)
            item_info = ItemInfo(item_key, "", image_size, bucket_reso, frame_count=frame_count, latent_cache_path=cache_file)
            item_info.text_encoder_output_cache_path = text_encoder_output_cache_file

            bucket = bucketed_item_info.get(bucket_reso, [])
            for _ in range(self.num_repeats):
                bucket.append(item_info)
            bucketed_item_info[bucket_reso] = bucket

        # prepare batch manager
        self.batch_manager = BucketBatchManager(bucketed_item_info, self.batch_size)
        self.batch_manager.show_bucket_info()

        self.num_train_items = sum([len(bucket) for bucket in bucketed_item_info.values()])

    def shuffle_buckets(self):
        # set random seed for this epoch
        random.seed(self.seed + self.current_epoch)
        self.batch_manager.shuffle()

    def __len__(self):
        if self.batch_manager is None:
            return 100  # dummy value
        return len(self.batch_manager)

    def __getitem__(self, idx):
        return self.batch_manager[idx]


class DatasetGroup(torch.utils.data.ConcatDataset):
    def __init__(self, datasets: Sequence[Union[ImageDataset, VideoDataset]]):
        super().__init__(datasets)
        self.datasets: list[Union[ImageDataset, VideoDataset]] = datasets
        self.num_train_items = 0
        for dataset in self.datasets:
            self.num_train_items += dataset.num_train_items

    def set_current_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_current_epoch(epoch)

    def set_current_step(self, step):
        for dataset in self.datasets:
            dataset.set_current_step(step)

    def set_max_train_steps(self, max_train_steps):
        for dataset in self.datasets:
            dataset.set_max_train_steps(max_train_steps)
