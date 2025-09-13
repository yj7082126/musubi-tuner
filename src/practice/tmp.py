import os,sys,glob
from tqdm import tqdm
import torch
from safetensors.torch import load_file, save_file

cache_directory = '/groups/chenchen/patrick/OpenS2V-Nexus/datasets/test3_part2_cache'
latent_cache_files = glob.glob(os.path.join(cache_directory, f"*_fp.safetensors"))
for latent_cache_path in tqdm(latent_cache_files):
    sd_latent = load_file(latent_cache_path)

    for k,v in sd_latent.items():
        if k.startswith("clean_latent_bboxes") and len(v.shape) == 3:
            sd_latent[k] = v[0]
        if k.startswith("clean_latent_indices"):
            if len(v.shape) == 0:
                sd_latent[k] = torch.tensor([v])
            elif v.shape[0] > 1:
                sd_latent[k] = v[[0]]

    save_file(sd_latent, latent_cache_path)