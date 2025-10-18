import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0,"
from pathlib import Path
import json, shutil
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
    lora_path = "outputs/training/idmask_control_lora_wrope_v2/idmask_control_lora_wrope_v2_5-step00006000.safetensors"
)
#%%
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# timestamp = '20251016_150910'
out_dir = Path("outputs/whisperer") / f"debug/en/{timestamp}"
out_dir.mkdir(parents=True, exist_ok=True)
seed = 48

main_layout_path = Path("/groups/chenchen/patrick/ViStoryBench/gen_layouts_bulk/20251004_200710_v2")
for story_num in tqdm(vistory_dataset.get_story_name_list()):
# for story_num in ['24']:
    (out_dir / story_num).mkdir(parents=True, exist_ok=True)
    (out_dir / f"{story_num}/debug").mkdir(parents=True, exist_ok=True)
    (out_dir / f"{story_num}/full").mkdir(parents=True, exist_ok=True)
    
    pose_layout = json.loads((main_layout_path / f"story_{story_num}/pose_layout.json").read_text())

    for shot_num, layout in pose_layout.items():
        shot_num = int(shot_num)
        # width, height = layout['canvas_size']['w'], layout['canvas_size']['h']
        width, height = 1344, 768
        panel_layout = {row['id']-1 : {
            'bbox' : list(map(lambda x: x/1000., [row['bbox']['x'], row['bbox']['y'], row['bbox']['x']+row['bbox']['w'], row['bbox']['y']+row['bbox']['h']])), 
            'body': []
        } for row in layout['boxes']}

        story_shot, characters_shot, prompt = get_info_from_vistorybench(vistory_dataset, story_num, shot_num)
        prompt = story_shot['type'] + ";" + prompt
        print(f"\n=== Story {story_num} - Shot {shot_num} ===")
        print(f"Prompt: {prompt}")

        result_imgs, debug_imgs, debug_mask = framepack_model(
            prompt, panel_layout, characters_shot, width, height,
            c_width_given=320, seed=seed, 
            debug_name=f"story{story_num}_shot{shot_num}"
        )

        result_imgs[0].save(out_dir / story_num / f"{shot_num-1}_0.png")
        debug_imgs[0].save(out_dir / story_num / f"debug/{shot_num-1}_0.png")

