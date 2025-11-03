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

sys.path.append("/groups/chenchen/patrick/dreambench_plus")
from dreambench_plus.dreambench_plus_dataset import DreamBenchPlus
dreambench_plus = DreamBenchPlus(dir="/groups/chenchen/patrick/dreambench_plus/data")

#%%
framepack_model = FramePack_1fmc(
    lora_path = "/home/yo564250/workspace/whisperer/related/framepackbase/musubi-tuner/outputs/training/idmask_control_lora_wrope_v3/idmask_control_lora_wrope_v3-3-step00006000.safetensors"
)
#%%
width, height = 1024, 1024
save_dir = Path("/groups/chenchen/patrick/dreambench_plus/samples")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
seed = 42

method_name = f"whisperer_{timestamp}_seed{seed}"
(save_dir / f"{method_name}/src_img").mkdir(parents=True, exist_ok=True)
(save_dir / f"{method_name}/tgt_img").mkdir(parents=True, exist_ok=True)
(save_dir / f"{method_name}/text").mkdir(parents=True, exist_ok=True)

for sample in tqdm(dreambench_plus):
    (save_dir / f"{method_name}/src_img/{sample.collection_id}").mkdir(parents=True, exist_ok=True)
    (save_dir / f"{method_name}/tgt_img/{sample.collection_id}").mkdir(parents=True, exist_ok=True)
    (save_dir / f"{method_name}/text/{sample.collection_id}").mkdir(parents=True, exist_ok=True)

    for caption_ind, prompt in enumerate(sample.captions):
        if sample.category == 'human':
            panel_layout = {0: {'bbox': [0.25, 0.25, 0.6, 0.9], 'body': []}}
            c_width = 320
            crop_face_detect = True
        else:
            panel_layout = {0: {'bbox': [0.25, 0.25, 0.75, 0.75], 'body': []}}
            c_width = 512
            crop_face_detect = False
        characters_shot = {0 : {'images' : [sample.image_path]}}

        result_imgs, debug_imgs, debug_mask = framepack_model(
            prompt, panel_layout, characters_shot, width, height,
            c_width_given=c_width, seed=seed, crop_face_detect=crop_face_detect, use_rembg=True,
            cache_results=False, cache_layers=['transformer_blocks.2'], 
            use_attention_masking=['no_cross_control_latents', 'mask_control'], 
            control_indices=[0], latent_indices=[3],
            debug_name=f"{sample.collection_id}-{caption_ind}"
        )

        sample.image.save(save_dir / f"{method_name}/src_img/{sample.collection_id}/{caption_ind}_0.jpg")
        result_imgs[0].convert("RGB").save(save_dir / f"{method_name}/tgt_img/{sample.collection_id}/{caption_ind}_0.jpg")
        (save_dir / f"{method_name}/text/{sample.collection_id}/{caption_ind}_0.txt").write_text(prompt)