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

sys.path.append("/home/yo564250/workspace/whisperer/main/evaluations/vistorybench")
# sys.path.append("/projects/bffz/ykwon4/vistorybench")
from vistorybench.data_process.dataset_process.dataset_load import StoryDataset
# vistory_dataset_path = Path("/lustre/fs1/groups/chenchen/patrick/ViStoryBench/dataset/ViStory")
vistory_dataset_path = Path("/home/yo564250/workspace/whisperer/main/evaluations/vistorybench/datasets/ViStory")
vistory_dataset = StoryDataset(vistory_dataset_path)

#%%
framepack_model = FramePack_1fmc(
    lora_path = "outputs/training/idmask_control_lora_wrope_v3/idmask_control_lora_wrope_v3-4-step00006000.safetensors"
    # dit_path = "/projects/bffz/ykwon4/ComfyUI/models/diffusion_models/FramePackI2V_HY_bf16.safetensors",
    # vae_path = "/projects/bffz/ykwon4/ComfyUI/models/vae/hunyuan-video-t2v-720p-vae.pt",
    # text_encoder1_path = "/projects/bffz/ykwon4/ComfyUI/models/text_encoders/llava_llama3_fp16.safetensors",
    # text_encoder2_path = "/projects/bffz/ykwon4/ComfyUI/models/text_encoders/clip_l.safetensors",
    # lora_path = "/work/nvme/bffz/ykwon4/musubi-training/idmask_control_lora_wrope_v3_2/test1-step00003000.safetensors"
)
#%%
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# timestamp = '20251102_085335'
width, height = 1280, 720

main_layout_path = Path("/groups/chenchen/patrick/ViStoryBench/gen_layouts_bulk/20250927_101053")
# main_layout_path = Path("/projects/bffz/ykwon4/vistorybench/data/gen_layouts_bulk/20251107_132846_v2")
out_dir = Path("/groups/chenchen/patrick/ViStoryBench/outputs/whisperer") / f"v3_4_step6000/en/{timestamp}"
# out_dir = Path("/projects/bffz/ykwon4/vistorybench/data/outputs/whisperer") / f"v3_multi_step5000/en/{timestamp}"
out_dir.mkdir(parents=True, exist_ok=True)
seed = 42

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
        prompt = ";".join(prompt.split(";")[:5])
        print(f"\n=== Story {story_num} - Shot {shot_num} ===")
        print(f"Prompt: {prompt}")

        layouts = [
            (x.stem, list(map(int, x.stem.split("-pages_")[-1].split("_")))) 
            for x in  list(main_layout_path.glob(f"story_{story_num}*"))]
        layout_name = [(x[0], x[1][0]) for x in layouts if shot_num in range(x[1][0], x[1][1]+1)][0]
        author_output_dir = main_layout_path / layout_name[0]

        layout = parse_bodylayout(author_output_dir / "pose_layout.json")
        panel_bbox, panel_layout = layout[f'[PANEL-{shot_num-layout_name[1]+1}]']
        width, height = getres(panel_bbox[2]-panel_bbox[0], panel_bbox[3]-panel_bbox[1], 
            target_area=1344*768, max_aspect_ratio=2.0)

        # if len(story_shot['character_key']) != len(panel_layout):
        #     if len(story_shot['character_key']) == 0:
        #         story_shot = vanila_shot
        #         characters_shot = vanila_character_shot
        #         panel_layout = {0: {'bbox': [0.1, 0.9, 0.2, 1.0], 'body': []}}
        #     elif len(story_shot['character_key']) == 1:
        #         panel_layout = random.choice([
        #             {0: {'bbox': [0.35, 0.2, 0.65, 1.0], 'body': []}}, 
        #             {0: {'bbox': [0.2, 0.3, 0.5, 1.0], 'body': []}}, 
        #             {0: {'bbox': [0.7, 0.3, 1.0, 1.0], 'body': []}}, 
        #         ])
        #     else:
        #         panel_layout = random.choice([
        #             {0: {'bbox': [0.2, 0.1, 0.5, 0.9], 'body': []}, 1: {'bbox': [0.5, 0.1, 0.9, 1.0], 'body': []}},
        #             {0: {'bbox': [0.1, 0.3, 0.4, 0.9], 'body': []}, 1: {'bbox': [0.5, 0.2, 0.9, 1.0], 'body': []}},
        #         ])

        result_imgs, debug_imgs, debug_mask = framepack_model(
            prompt, panel_layout, characters_shot, width, height,
            c_width_given=320, seed=seed, crop_face_detect=True, use_rembg=True,
            use_attention_masking=['no_cross_control_latents'], 
            debug_name=f"story{story_num}_shot{shot_num}"
        )
        result_imgs[0].save(out_dir / story_num / f"{shot_num-1}_0.png")
        debug_imgs[0].save(out_dir / story_num / f"debug/{shot_num-1}_0.png")

