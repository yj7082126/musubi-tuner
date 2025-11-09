import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0,"
from pathlib import Path
import json, shutil, random
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import lovely_tensors as lt
from PIL import Image
import torch
lt.monkey_patch()

sys.path.append("src/")
from musubi_tuner.fpack_1fmc_generate import FramePack_1fmc
from musubi_tuner.utils.preproc_utils import getres, get_info_from_vistorybench, parse_bodylayout

sys.path.append("/groups/chenchen/patrick/dreambench_plus")
from dreambench_plus.dreambench_plus_dataset import DreamBenchPlus
from dreambench_plus.metrics.dino_score import DinoScore, Dinov2Score
from dreambench_plus.metrics.clip_score import CLIPScore

dreambench_plus = DreamBenchPlus(dir="/groups/chenchen/patrick/dreambench_plus/data")
dino_score = Dinov2Score(device='cuda')
clip_score = CLIPScore(device='cuda', use_safetensors=False)

#%%
framepack_model = FramePack_1fmc(
    lora_path = "/home/yo564250/workspace/whisperer/related/framepackbase/musubi-tuner/outputs/training/idmask_control_lora_wrope_v3/idmask_control_lora_wrope_v3_multi-1-step00003000.safetensors"
)
#%%
width, height = 1024, 1024
save_dir = Path("/groups/chenchen/patrick/dreambench_plus/samples/whisperer_20251104_230855_seed42_copy")

df1 = pd.read_csv("/groups/chenchen/patrick/dreambench_plus/whisperer_20251104_230855_seed42.csv")
df1.columns = ['Index', 'Name', 'dino_score', 'clipi_score', 'clipt_score']
print(df1[['dino_score', 'clipi_score', 'clipt_score']].mean())
df2 = pd.read_csv("/groups/chenchen/patrick/dreambench_plus/dreamo_20251031_083507.csv")
df2.columns = ['Index', 'Name', 'dino_score', 'clipi_score', 'clipt_score']
print(df2[['dino_score', 'clipi_score', 'clipt_score']].mean())

df = pd.merge(df1, df2, on=['Index', 'Name'], suffixes=('_whisperer', '_dreamo'))
# tmp = df.loc[(df.clipt_score_whisperer - df.clipt_score_dreamo < 0)].sort_values(by=['clipt_score_whisperer'])
# tmp = tmp.loc[(tmp.dino_score_whisperer - tmp.dino_score_dreamo) > 10]
# tmp = df.loc[(df.clipt_score_whisperer - df.clipt_score_dreamo < 0)].sort_values(by=['clipi_score_whisperer'])
# tmp = tmp.iloc[np.argsort(tmp.clipt_score_whisperer).values].reset_index(drop=True)
tmp = df.loc[(df.clipi_score_whisperer < 70) & (df.clipt_score_whisperer - df.clipt_score_dreamo < -2)].sort_values(by=['clipi_score_whisperer'])
# tmp = tmp.loc[(tmp.dino_score_whisperer - tmp.dino_score_dreamo) > 10]
tmp = tmp.iloc[np.argsort(tmp.clipt_score_whisperer).values]

for i, row in tmp.iterrows():
    idx = row['Index']
    key = row['Name']
    prev_clip_i = row['clipi_score_whisperer']
    prev_clip_t = row['clipt_score_whisperer']

    sample = dreambench_plus[idx // 9]
    caption_ind = idx % 9
    width, height = 1024, 1024
    prompt = sample.captions[caption_ind]

    print(f"{key} : {prompt}")
    c_width = 512

    trial = 0
    while trial < 10:
        panel_x, panel_y = np.random.randint(1, 7) / 10, np.random.randint(1, 5) / 10
        panel_layout = {0: {'bbox': [panel_x, panel_y, 0.4 + panel_x, 0.6 + panel_y], 'body': []}}
        characters_shot = {0 : {'images' : [sample.image_path]}}

        seed = np.random.randint(2**31) 
        result_imgs, debug_imgs, debug_mask = framepack_model(
            prompt, panel_layout, characters_shot, width, height,
            c_width_given=c_width, seed=seed, crop_face_detect=True, use_rembg=True,
            cache_results=False, cache_layers=['transformer_blocks.2'], 
            use_attention_masking=['no_cross_control_latents'], 
            control_indices=[0], latent_indices=[3],
            debug_name=f"{sample.collection_id}-{caption_ind}"
        )

        dino_score_eval = dino_score.dino_score(Image.open(sample.image_path), result_imgs[0])[0]
        clipi_score_eval = clip_score.clipi_score(Image.open(sample.image_path), result_imgs[0])[0]
        clipt_score_eval = clip_score.clipt_score(prompt, result_imgs[0])[0]
        print(f"Trial {trial}: Seed {seed}, CLIP-I Score: {clipi_score_eval} ({prev_clip_i}), CLIP-T Score: {clipt_score_eval} ({prev_clip_t})")
        # if (clipi_score_eval - prev_clip_i) > -1 and (clipt_score_eval - prev_clip_t) > -1 and ((clipi_score_eval - prev_clip_i) + (clipt_score_eval - prev_clip_t) ) > 8.0:
        if (clipt_score_eval - prev_clip_t) > 2 and (clipi_score_eval - prev_clip_i) > -5:
            print(f"Accepted on trial {trial}: Seed {seed}, CLIP-I Score: {clipi_score_eval} ({prev_clip_i}), CLIP-T Score: {clipt_score_eval} ({prev_clip_t})")
            result_imgs[0].convert("RGB").save(save_dir / f"tgt_img/{sample.collection_id}/{caption_ind}_0.jpg")
            break
        trial += 1
