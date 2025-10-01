import os, sys
sys.path.append("src")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,"
from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import lovely_tensors as lt
from einops import rearrange
from PIL import Image, ImageDraw, ImageOps
import torch
lt.monkey_patch()
from safetensors.torch import load_file

from musubi_tuner.networks import lora_framepack
from musubi_tuner.frame_pack.framepack_utils import load_vae, load_text_encoder1, load_text_encoder2, load_image_encoders
from musubi_tuner.frame_pack.hunyuan_video_packed import load_packed_model, attn_cache
from musubi_tuner.frame_pack.k_diffusion_hunyuan import sample_hunyuan
from musubi_tuner.wan_generate_video import merge_lora_weights
from musubi_tuner.utils.preproc_utils import preproc_image, preproc_mask, postproc_imgs
from musubi_tuner.utils.preproc_utils import get_text_preproc, prepare_control_inputs_for_entity, get_all_kwargs_from_opens2v_metapath
from musubi_tuner.utils.preproc_utils import getres, get_info_from_vistorybench, parse_bodylayout
from musubi_tuner.utils.attn_utils import get_pltplot_as_pil, get_text_inds_from_dict, get_attn_map
from musubi_tuner.utils.viz_utils import printable_metadata, return_total_visualization
from musubi_tuner.utils.bbox_utils import get_mask_from_bboxes, draw_bboxes, draw_bboxes_images, auto_scale_layout_data, get_bbox_from_mask
from musubi_tuner.utils.keypalign_utils import search_facebbox_for_layout

# sys.path.append("/lustre/fs1/home/yo564250/workspace/whisperer/datasets/storyviz/vistorybench")
sys.path.append("/projects/bffz/ykwon4/vistorybench/")
from vistorybench.data_process.dataset_process.dataset_load import StoryDataset

device = torch.device('cuda')
#%%
main_path = Path("/projects/bffz/ykwon4/ComfyUI/models")
# main_path = Path("/lustre/fs1/home/yo564250/workspace/ComfyUI/models")
dit_path = "diffusion_models/FramePackI2V_HY_bf16.safetensors"
vae_path = "vae/hunyuan-video-t2v-720p-vae.pt"
text_encoder1_path = "text_encoders/llava_llama3_fp16.safetensors"
text_encoder2_path = "text_encoders/clip_l.safetensors"
# lora_path = '../../outputs/training/idmask_control_lora_wrope_v2/idmask_control_lora_wrope_v2_4-step00006000.safetensors'
lora_path = main_path / "loras/idmask_control_lora_wrope_v2_4-step00006000.safetensors"

model = load_packed_model(device, main_path / dit_path, 'sageattn', device, has_image_proj=False)
model.to(device)
model.eval().requires_grad_(False)

if lora_path is not None:
    merge_lora_weights(lora_framepack, model, 
        SimpleNamespace(
            lora_weight = [lora_path], lora_multiplier = [1.0], 
            include_patterns=None, exclude_patterns=None, lycoris=None, save_merged_model=False
        ), 
    device, None)

vae = load_vae(str(main_path / vae_path), 32, 128, device)

tokenizer1, text_encoder1 = load_text_encoder1(SimpleNamespace(text_encoder1=str(main_path / text_encoder1_path)), False, device)
tokenizer2, text_encoder2 = load_text_encoder2(SimpleNamespace(text_encoder2=str(main_path / text_encoder2_path)))

#%%
def get_control_kwargs_full(panel_layout, characters_shot, width, height, 
        crop_face_detect=True, use_face_detect=True,
        c_width_given=None, scale_c=1.2, use_safety=True,
        bbox_mode="full_width_relative_height",
        max_characters=2, control_indices=[0], latent_indices=[3], 
        use_rembg=True):
    auto_scaled_layout, metadata = auto_scale_layout_data(panel_layout)

    debug_dict = search_facebbox_for_layout(
        auto_scaled_layout, characters_shot, (width, height), 
        crop_face_detect=crop_face_detect, use_face_detect=use_face_detect,
        c_width_given=c_width_given, scale_c=scale_c, use_safety=use_safety,
        bbox_mode=bbox_mode)

    print_res = ""
    for k,v in debug_dict.items():
        print_res += f"Entity {k+1} (Use Crop: {crop_face_detect})\n"
        print_res += f"\tControl Image Path: {v['control_image_path']}\n"
        print_res += f"\tControl Image Size: {v['control_image_size']}\n"
        print_res += "\tAttn BBox: [" + ', '.join([f"{b:.3f}" for b in v['entity_bbox']]) + "]\n"
        print_res += "\tFace BBox: [" + ', '.join([f"{b:.3f}" for b in v['face_bbox']]) + "]\n"

    n_chara = min(len(debug_dict), len(characters_shot), max_characters)
    if len(debug_dict) == 0:
        control_images = []
        control_image_sizes = []
        entity_bboxes = []
        face_bboxes = []
        entity_masks = None
        debug_mask = Image.new("RGB", (width, height), (0,0,0))
    else:
        control_images = [debug_dict[i]['control_image'] for i in range(n_chara)]
        control_image_sizes = [debug_dict[i]['control_image_size'] for i in range(n_chara)]
        entity_bboxes = [debug_dict[i]['entity_bbox'] for i in range(n_chara)]
        face_bboxes = [debug_dict[i]['face_bbox'] for i in range(n_chara)]
        entitymask_nps = [get_mask_from_bboxes([entity_bbox], width, height) for entity_bbox in entity_bboxes]
        entity_masks = torch.cat([preproc_mask(e_mask, width, height, invert=False)[0] for e_mask in entitymask_nps], 2)

        debug_mask = Image.fromarray(np.sum(entitymask_nps, axis=0)>0).convert("RGB")
        debug_mask = draw_bboxes_images(debug_mask, face_bboxes, control_images, cimg_sizes=control_image_sizes)
        debug_mask = draw_bboxes(debug_mask, face_bboxes, width=4)

    control_kwargs, control_nps = prepare_control_inputs_for_entity(
        control_images, entity_bboxes, width, height, vae,
        control_image_sizes,
        face_entity_bboxes=face_bboxes,
        control_indices=control_indices, latent_indices=latent_indices,
        adjust_custom_wh=False, 
        mode="provided_face_bbox",  # mode="provided_size_mid_x",
        use_rembg=use_rembg)
    if len(control_nps) > 0:
        control_nps = np.concatenate([
            np.asarray(Image.fromarray(x).resize((256,256))) for x in control_nps], 
        axis=1)
    else:
        control_nps = np.zeros((256,256,3), dtype=np.uint8)
    return control_kwargs, entity_masks, debug_mask, control_nps, print_res

#%%
vistory_dataset_path = Path('/projects/bffz/ykwon4/vistorybench/data/dataset/ViStory')
vistory_dataset = StoryDataset(vistory_dataset_path)
# main_layout_path = Path(f"/groups/chenchen/patrick/ViStoryBench/gen_layouts_bulk/20250927_101053/")
main_layout_path = Path("/projects/bffz/ykwon4/vistorybench/data/gen_layouts_bulk/20250927_101053/")
# currtime = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
currtime = '20250930_110059'
# main_output_path = Path(f"/home/yo564250/workspace/whisperer/related/framepackbase/musubi-tuner/outputs/vistory_test/base/en/{currtime}")
main_output_path = Path(f"/projects/bffz/ykwon4/vistorybench/data/outputs/whisperer/base/en/{currtime}")
main_output_path.mkdir(parents=True, exist_ok=True)
debug_output_path = Path(f"/projects/bffz/ykwon4/vistorybench/data/outputs/whisperer_debug/base/en/{currtime}")
debug_output_path.mkdir(parents=True, exist_ok=True)

#%%
max_scene_sentences = 3
max_characters = 2
scale_c = 1.2

for story_num in [f"{x:02d}" for x in range(40,81)]:
    shot_nums = [x['index'] for x in vistory_dataset.load_shots(story_num)]
    output_path = main_output_path / f"{story_num}"
    output_path.mkdir(parents=True, exist_ok=True)
    d_output_path = debug_output_path / f"{story_num}"
    d_output_path.mkdir(parents=True, exist_ok=True)
    for shot_num in shot_nums:
        # try:
        if (output_path / f"{shot_num-1}_0.png").is_file():
            continue
        print(f"Processing for story {story_num} shot {shot_num}")

        story_shot, characters_shot = get_info_from_vistorybench(vistory_dataset, story_num, shot_num)
        prompt = story_shot['type'] + ". " + ", ".join(story_shot['scene'].split(", ")[:max_scene_sentences]) + ". " + story_shot['script']
        text_kwargs = get_text_preproc(prompt, 
            text_encoder1, text_encoder2, tokenizer1, tokenizer2, 
            entity_prompts=[], device=device)

        layouts = [
            (x.stem, list(map(int, x.stem.split("-pages_")[-1].split("_")))) 
            for x in  list(main_layout_path.glob(f"story_{story_num}*"))]
        layout_name = [(x[0], x[1][0]) for x in layouts if shot_num in range(x[1][0], x[1][1]+1)][0]
        author_output_dir = main_layout_path / layout_name[0]
        layout = parse_bodylayout(author_output_dir / "pose_layout.json")

        panel_bbox, panel_layout = layout[f'[PANEL-{shot_num-layout_name[1]+1}]']
        width, height = getres(panel_bbox[2]-panel_bbox[0], panel_bbox[3]-panel_bbox[1], 
            target_area=1280*720, max_aspect_ratio=2.0)
        
        control_kwargs, entity_masks, debug_mask, control_nps, print_res = get_control_kwargs_full(
            panel_layout, characters_shot, width, height, 
            crop_face_detect=True, use_face_detect=True, c_width_given=None, scale_c=scale_c,
            max_characters=max_characters, control_indices=[0], latent_indices=[3], 
            use_rembg=True
        )
        print(print_res)

        seed = np.random.randint(2**31)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        total_kwargs = {
            'prompt': prompt, 'sampler': 'unipc', 'width': width, 'height': height, 'frames': 1, 'batch_size': 1,
            'num_inference_step': 25, 'generator': generator,
            'device': device, 'dtype': torch.bfloat16,
            'cache_results': False, 'cache_layers': [], 
            'use_attention_masking': ['no_cross_control_latents', 'mask_control'],
            'entity_masks': entity_masks,
        }
        attn_cache.clear()
        results = sample_hunyuan(transformer=model, **total_kwargs, **text_kwargs, **control_kwargs,)
        result_img = Image.fromarray(postproc_imgs(results,vae)[0])

        meta_str = printable_metadata(total_kwargs, text_kwargs, control_kwargs, lora_path, maxlen=80, seed=seed)
        meta_str_2 = meta_str + "\n\n" + print_res
        attn_mask = get_pltplot_as_pil(attn_cache['attn_mask'][0], vmin=-9999., vmax=0., cmap=plt.cm.viridis)
        debug_img = return_total_visualization(
            f'story{story_num}_shot{shot_num}_seed{seed}', 
            meta_str, np.asarray(result_img), 
            attn_mask, np.asarray(control_nps), np.asarray(debug_mask), 
            np.zeros((height, width, 3), dtype=np.uint8))

        result_img.save(output_path / f"{shot_num-1}_0.png")
        debug_img.save(d_output_path / f"{shot_num-1}_debug.png")
        debug_mask.save(d_output_path / f"{shot_num-1}_mask.png")
        try:
            (d_output_path / f'{shot_num-1}_meta.txt').write_text(meta_str_2, encoding='cp949')
        except Exception as e:
            print(f"Error in loading data for story {story_num} shot {shot_num}: {e}")
            continue
        # except Exception as e:
        #     print(f"Error in loading data for story {story_num} shot {shot_num}: {e}")
        #     continue
