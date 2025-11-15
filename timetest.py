import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0,"
from pathlib import Path
import json, shutil, random
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf
import numpy as np
import lovely_tensors as lt
from PIL import Image
import torch
lt.monkey_patch()

sys.path.append("src/")
from musubi_tuner.fpack_1fmc_generate import FramePack_1fmc
from musubi_tuner.utils.preproc_utils import getres, get_info_from_vistorybench, parse_bodylayout

sys.path.append("/lustre/fs1/home/yo564250/workspace/whisperer/datasets/storyviz/vistorybench")
# sys.path.append("/projects/bffz/ykwon4/vistorybench")
from vistorybench.data_process.dataset_process.dataset_load import StoryDataset
vistory_dataset_path = Path("/lustre/fs1/groups/chenchen/patrick/ViStoryBench/dataset/ViStory")
# vistory_dataset_path = Path("/projects/bffz/ykwon4/vistorybench/data/dataset/ViStory")
vistory_dataset = StoryDataset(vistory_dataset_path)
import csv
import time
from datetime import datetime
try:
    import pynvml
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

#%%
framepack_model = FramePack_1fmc(
    lora_path = "/home/yo564250/workspace/whisperer/related/framepackbase/musubi-tuner/outputs/training/idmask_control_lora_wrope_v3/idmask_control_lora_wrope_v3_multi-1-step00005000.safetensors"
)

#%% ------------------------------- GPU Utils ---------------------------------- #
def init_nvml():
    if _HAS_NVML:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return handle
    return None


def get_nvml_stats(handle):
    """
    Returns dict with NVML memory + utilization for GPU 0.
    If NVML unavailable, returns Nones.
    """
    if handle is None:
        return {
            "nvml_mem_used_MB": None,
            "nvml_mem_total_MB": None,
            "nvml_util_percent": None,
        }
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return {
        "nvml_mem_used_MB": mem.used / (1024 ** 2),
        "nvml_mem_total_MB": mem.total / (1024 ** 2),
        "nvml_util_percent": util.gpu,
    }


def get_torch_mem():
    """
    Returns current and peak memory stats from PyTorch CUDA.
    """
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_alloc = torch.cuda.max_memory_allocated()
    return {
        "torch_mem_allocated_MB": allocated / 1e6,
        "torch_mem_reserved_MB": reserved / 1e6,
        "torch_max_mem_allocated_MB": max_alloc / 1e6,
    }

nvml_handle = init_nvml()
fieldnames = [
    "run_idx",
    "wall_time_sec",
    "torch_mem_allocated_MB",
    "torch_mem_reserved_MB",
    "torch_max_mem_allocated_MB",
    "nvml_mem_used_MB",
    "nvml_mem_total_MB",
    "nvml_gpu_util_percent",
    "timestamp",
]


#%%
main_layout_path = Path("/groups/chenchen/patrick/ViStoryBench/gen_layouts_bulk/20251107_132846_v2")
story_num = '04'
shot_num = 4
width, height = 1280, 720

story_shot, characters_shot, prompt = get_info_from_vistorybench(vistory_dataset, story_num, shot_num)
prompt = ";".join(prompt.split(";")[:5])

layouts = [
    (x.stem, list(map(int, x.stem.split("-pages_")[-1].split("_")))) 
    for x in  list(main_layout_path.glob(f"story_{story_num}*"))]
layout_name = [(x[0], x[1][0]) for x in layouts if shot_num in range(x[1][0], x[1][1]+1)][0]
author_output_dir = main_layout_path / layout_name[0]

layout = parse_bodylayout(author_output_dir / "pose_layout.json")
panel_bbox, panel_layout = layout[f'[PANEL-{shot_num-layout_name[1]+1}]']

def generate_image(framepack_model, seed=42):
    result_imgs, debug_imgs, debug_mask = framepack_model(
        prompt, panel_layout, characters_shot, width, height,
        c_width_given=1024, seed=seed, crop_face_detect=True, use_rembg=True,
        cache_results=True, cache_layers=['transformer_blocks.2'], 
        use_attention_masking=['no_cross_control_latents'], 
        control_indices=[0], latent_indices=[3], max_chara=5,
        debug_name=f"story{story_num}_shot{shot_num}"
    )
    return result_imgs[0]

for _ in range(3):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = generate_image(framepack_model, seed=42)
    torch.cuda.synchronize()

rows = []
os.makedirs("generated", exist_ok=True)

for i in range(1, 10 + 1):
    seed = 1000 + i

    # Reset and sync before timing
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    img = generate_image(framepack_model, seed=seed)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    img.save(os.path.join("generated", f"sample_{i:02d}.png"))

    # Collect metrics
    torch_stats = get_torch_mem()
    nvml_stats = get_nvml_stats(nvml_handle)
    ts = datetime.now().isoformat(timespec="seconds")

    row = {
        "run_idx": i,
        "wall_time_sec": round(wall, 4),
        **{k: round(v, 2) if isinstance(v, (int, float)) and v is not None else v
            for k, v in torch_stats.items()},
        "nvml_mem_used_MB": None if nvml_stats["nvml_mem_used_MB"] is None else round(nvml_stats["nvml_mem_used_MB"], 2),
        "nvml_mem_total_MB": None if nvml_stats["nvml_mem_total_MB"] is None else round(nvml_stats["nvml_mem_total_MB"], 2),
        "nvml_gpu_util_percent": None if nvml_stats["nvml_util_percent"] is None else int(nvml_stats["nvml_util_percent"]),
        "timestamp": ts,
    }
    rows.append(row)
    print(f"[Run {i:02d}] {row}")

# Write CSV
with open('dreamcomic_1280x720_1024.csv', "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved metrics to: dreamcomic_1280x720_1024.csv")
if _HAS_NVML:
    pynvml.nvmlShutdown()
