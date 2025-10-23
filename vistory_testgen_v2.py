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
from musubi_tuner.utils.preproc_utils import get_info_from_vistorybench

sys.path.append("/lustre/fs1/home/yo564250/workspace/whisperer/datasets/storyviz/vistorybench")
from vistorybench.data_process.dataset_process.dataset_load import StoryDataset
vistory_dataset_path = Path("/lustre/fs1/groups/chenchen/patrick/ViStoryBench/dataset/ViStory")
vistory_dataset = StoryDataset(vistory_dataset_path)

#%%
framepack_model = FramePack_1fmc(
    lora_path = "outputs/training/idmask_control_lora_wrope_v2_multi/idmask_control_lora_wrope_v2_multi_2-step00003000.safetensors"
)
#%%
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# timestamp = '20251016_150910'
width, height = 1344, 768
out_dir = Path("/lustre/fs1/groups/chenchen/patrick/ViStoryBench/outputs/whisperer") / f"debug/en/{timestamp}"
out_dir.mkdir(parents=True, exist_ok=True)
seed = 2048

for story_num in tqdm(vistory_dataset.get_story_name_list()):
    (out_dir / story_num).mkdir(parents=True, exist_ok=True)
    (out_dir / f"{story_num}/debug").mkdir(parents=True, exist_ok=True)

    story = vistory_dataset.load_story(story_num)
    characters = vistory_dataset.load_characters(story_num)
    story_dict = {x['index']:x for x in story['shots']}

    vanila_shot, vanila_character_shot, _ = get_info_from_vistorybench(vistory_dataset, story_num, 1)
    for shot_num, story_shot in story_dict.items():
        shot_num = int(shot_num)
        story_shot, characters_shot, prompt = get_info_from_vistorybench(vistory_dataset, story_num, shot_num)
        prompt = story_shot['type'] + ";" + prompt
        print(f"\n=== Story {story_num} - Shot {shot_num} ===")
        print(f"Prompt: {prompt}")

        if len(story_shot['character_key']) == 0:
            story_shot = vanila_shot
            characters_shot = vanila_character_shot
            panel_layout = {0: {'bbox': [0.1, 0.9, 0.2, 1.0], 'body': []}}
        elif len(story_shot['character_key']) == 1:
            panel_layout = random.choice([
                {0: {'bbox': [0.35, 0.2, 0.65, 1.0], 'body': []}}, 
                {0: {'bbox': [0.2, 0.3, 0.5, 1.0], 'body': []}}, 
                {0: {'bbox': [0.7, 0.3, 1.0, 1.0], 'body': []}}, 
            ])
        else:
            panel_layout = random.choice([
                {0: {'bbox': [0.2, 0.1, 0.5, 0.9], 'body': []}, 1: {'bbox': [0.5, 0.1, 0.9, 1.0], 'body': []}},
                {0: {'bbox': [0.1, 0.3, 0.4, 0.9], 'body': []}, 1: {'bbox': [0.5, 0.2, 0.9, 1.0], 'body': []}},
            ])

        result_imgs, debug_imgs, debug_mask = framepack_model(
            prompt, panel_layout, characters_shot, width, height,
            c_width_given=400, seed=seed, crop_face_detect=True, use_rembg=True,
            debug_name=f"story{story_num}_shot{shot_num}"
        )
        result_imgs[0].save(out_dir / story_num / f"{shot_num-1}_0.png")
        debug_imgs[0].save(out_dir / story_num / f"debug/{shot_num-1}_0.png")

