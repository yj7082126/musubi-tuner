"""Concise Gradio demo for FramePack Hunyuan single-image generation based on first 3 cells of test_seg_v3.ipynb.
Adjust the model paths below to match your local ComfyUI model directory if different.
"""
from doctest import debug
import os, sys, random
sys.path.append("src")
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import lovely_tensors as lt
from PIL import Image
import torch
import gradio as gr
lt.monkey_patch()

#%%
# ---------------------------------------------------------------------------
# Configuration (edit these to match your environment)
# ---------------------------------------------------------------------------
MAIN_PATH = Path(os.environ.get("COMFY_MODELS", "/lustre/fs1/home/yo564250/workspace/ComfyUI/models"))
DIT_MODEL = "diffusion_models/FramePackI2V_HY_bf16.safetensors"
VAE_PATH = "vae/hunyuan-video-t2v-720p-vae.pt"
TEXT_ENCODER1 = "text_encoders/llava_llama3_fp16.safetensors"
TEXT_ENCODER2 = "text_encoders/clip_l.safetensors"
IMAGE_ENCODER = "clip_vision/sigclip_vision_patch14_384.safetensors"

LORA_DIR = 'outputs/training'  # directory with LoRA models
LORA_PATH = None  # set to None to disable
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

#%%
# ---------------------------------------------------------------------------
# Lazy globals (loaded once)
# ---------------------------------------------------------------------------
model = None
curr_lora_path = LORA_PATH
vae = None
tokenizer1 = tokenizer2 = text_encoder1 = text_encoder2 = None

# Cache for attention (mirroring notebook usage)
try:
    from musubi_tuner.frame_pack.hunyuan_video_packed import load_packed_model, attn_cache
    from musubi_tuner.frame_pack.framepack_utils import load_vae, load_text_encoder1, load_text_encoder2
    from musubi_tuner.frame_pack.k_diffusion_hunyuan import sample_hunyuan
    from musubi_tuner.networks import lora_framepack
    from musubi_tuner.wan_generate_video import merge_lora_weights
    from musubi_tuner.utils.bbox_utils import get_mask_from_bboxes, draw_bboxes, get_bbox_from_str, draw_bboxes_images
    from musubi_tuner.utils.bbox_utils import auto_scale_layout_data, get_bbox_from_mask
    from musubi_tuner.utils.keypalign_utils import search_facebbox_for_layout
    from musubi_tuner.utils.preproc_utils import preproc_mask, postproc_imgs, get_text_preproc, prepare_control_inputs_for_entity
except ImportError as e:
    raise SystemExit(f"Required project modules not found: {e}. Make sure you run from project root.")

#%%

def load_pipeline(lora_path=LORA_PATH, lora_multiplier=1.0):
    """Load model, VAE, encoders and (optionally) merge LoRA once."""
    global model, curr_lora_path, vae, tokenizer1, tokenizer2, text_encoder1, text_encoder2
    if model is not None or curr_lora_path == lora_path:
        return
    else:
        print(f"Loading model with lora {lora_path}...")

    model = load_packed_model(DEVICE, MAIN_PATH / DIT_MODEL, 'sageattn', DEVICE, has_image_proj=False)
    model.to(DEVICE)
    model.eval().requires_grad_(False)
    if Path(os.path.join(LORA_DIR, lora_path)).exists():
        merge_lora_weights(lora_framepack, model, SimpleNamespace(
            lora_weight=[os.path.join(LORA_DIR, lora_path)], lora_multiplier=[lora_multiplier],
            include_patterns=None, exclude_patterns=None,
            lycoris=None, save_merged_model=False
        ), DEVICE, None)
        curr_lora_path = lora_path

    # Load auxiliaries
    vae = load_vae(str(MAIN_PATH / VAE_PATH), 32, 128, DEVICE)
    tokenizer1, text_encoder1 = load_text_encoder1(SimpleNamespace(text_encoder1=str(MAIN_PATH / TEXT_ENCODER1)), False, DEVICE)
    tokenizer2, text_encoder2 = load_text_encoder2(SimpleNamespace(text_encoder2=str(MAIN_PATH / TEXT_ENCODER2)))

#%%
def get_control_kwargs_full(panel_layout, characters_shot, width, height, 
        crop_face_detect=True, use_face_detect=True,
        c_width_given=None, scale_c=1.5, use_safety=True,
        bbox_mode="full_width_relative_height",
        max_characters=2, control_indices=[0], latent_indices=[3], 
        use_rembg=True, print_res=False):
    
    auto_scaled_layout, metadata = auto_scale_layout_data(panel_layout)
    print(characters_shot)
    debug_dict = search_facebbox_for_layout(
        auto_scaled_layout, characters_shot, (width, height), 
        crop_face_detect=crop_face_detect, use_face_detect=use_face_detect,
        c_width_given=c_width_given, bbox_mode=bbox_mode,
        scale_c=scale_c, use_safety=use_safety,
    )
    
    if print_res:
        for k,v in debug_dict.items():
            print(f"Entity {k+1}")
            print(f"\tControl Image Path: {v['control_image_path']}")
            print(f"\tControl Image Size: {v['control_image_size']}")
            print("\tAttn BBox: [" + ', '.join([f"{b:.3f}" for b in v['entity_bbox']]) + "]")
            print("\tFace BBox: [" + ', '.join([f"{b:.3f}" for b in v['face_bbox']]) + "]")

    n_chara = min(len(debug_dict), len(characters_shot), max_characters)
    if len(debug_dict) == 0:
        control_image_paths = []
        control_image_sizes = []
        entity_bboxes = []
        face_bboxes = []
        entity_masks = None
        debug_mask = Image.new("RGB", (width, height), (0,0,0))
    else:
        control_image_paths = [debug_dict[i]['control_image_path'] for i in range(n_chara)]
        control_image_sizes = [debug_dict[i]['control_image_size'] for i in range(n_chara)]
        entity_bboxes = [debug_dict[i]['entity_bbox'] for i in range(n_chara)]
        face_bboxes = [debug_dict[i]['face_bbox'] for i in range(n_chara)]
        entitymask_nps = [get_mask_from_bboxes([entity_bbox], width, height) for entity_bbox in entity_bboxes]
        entity_masks = torch.cat([preproc_mask(e_mask, width, height, invert=False)[0] for e_mask in entitymask_nps], 2)

        debug_mask = Image.fromarray(np.sum(entitymask_nps, axis=0)>0).convert("RGB")
        debug_mask = draw_bboxes_images(debug_mask, face_bboxes, control_image_paths, cimg_sizes=control_image_sizes)
        debug_mask = draw_bboxes(debug_mask, face_bboxes, width=4)

    control_kwargs, control_nps = prepare_control_inputs_for_entity(
        control_image_paths, entity_bboxes, width, height, vae,
        control_image_sizes, face_entity_bboxes=face_bboxes,
        control_indices=[0], latent_indices=[3],
        adjust_custom_wh=False, 
        mode="provided_face_bbox",  # mode="provided_size_mid_x",
        use_rembg=True)
    if len(control_nps) > 0:
        control_nps = np.concatenate([
            np.asarray(Image.fromarray(x).resize((256,256))) for x in control_nps], 
        axis=1)
    else:
        control_nps = np.zeros((256,256,3), dtype=np.uint8)

    control_kwargs, control_nps = prepare_control_inputs_for_entity(
        control_image_paths, entity_bboxes, width, height, vae,
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
    return control_kwargs, entity_masks, debug_mask, control_nps

#%%
def generate_image(lora_path: str, prompt: str, bbox_str: str, control_image: Image.Image | None,
                   width: int, height: int, 
                   steps: int = 25, guidance: float = 1.0, seed: int | None = -1, 
                   target_index: int = 3, control_index: int = 0, 
                   control_mode: str = "provided_size_mid_x",
                   scale_c: float = 1.2, c_width_given: int | None = None,
                   use_rembg: bool = False, use_facecrop: bool = True
                   ):
    load_pipeline(lora_path=lora_path)

    if seed is None or seed < 0:
        seed = random.randint(0, 2**31 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Parse bbox string "x1,y1,x2,y2" normalized (0-1) or absolute pixels
    entity_bboxes = [get_bbox_from_str(bbox_str, width=width, height=height)]
    layout_dict = {i : {'bbox' : v, 'body': []} for i,v in enumerate(entity_bboxes)}

    # Save control image if provided, else create white placeholder for same flow
    control_image_paths = []
    if control_image is not None:
        tmp_path = Path("outputs/tmp/gradio_control.png")
        control_image.convert("RGB").save(tmp_path)
        control_image_paths.append(str(tmp_path))
    else:
        # Use a white placeholder if none given
        tmp_path = Path("outputs/tmp/gradio_control_white.png")
        Image.new("RGB", (width, height), (255,255,255)).save(tmp_path)
        control_image_paths.append(str(tmp_path))
    characters_shot = {i : {'images' : [v]} for i,v in enumerate(control_image_paths)}

    # Build entity mask
    # entitymask_nps = [get_mask_from_bboxes(entity_bboxes, width, height)]
    # entity_masks = torch.cat([preproc_mask(e_mask, width, height, invert=False)[0] for e_mask in entitymask_nps], 1)

    # Text preprocessing
    text_kwargs = get_text_preproc(prompt, text_encoder1, text_encoder2, tokenizer1, tokenizer2, entity_prompts=[], device=DEVICE)

    c_width_given = None if c_width_given <= 0 else c_width_given
    control_kwargs, entity_masks, debug_mask, control_nps = get_control_kwargs_full(
        layout_dict, characters_shot, width, height, 
        crop_face_detect=use_facecrop, use_face_detect=True,
        c_width_given=c_width_given, scale_c=scale_c, use_safety=True,
        bbox_mode=control_mode, max_characters=1, 
        control_indices=[control_index], latent_indices=[target_index], 
        use_rembg=use_rembg, print_res=True)

    total_kwargs = {
        'prompt': prompt, 'sampler': 'unipc', 'width': width, 'height': height, 'frames': 1,
        'batch_size': 1, 'real_guidance_scale': 1.0, 'distilled_guidance_scale': guidance,
        'guidance_rescale': 0.0, 'shift': None, 'num_inference_step': steps,
        'generator': generator, 'device': DEVICE, 'dtype': DTYPE,
        'cache_results': True, 'cache_layers': [],
        'use_attention_masking': ['mask_control'], 'entity_masks': entity_masks,
    }
    attn_cache.clear()
    with torch.inference_mode():
        results = sample_hunyuan(transformer=model, **total_kwargs, **text_kwargs, **control_kwargs)
    result_img = Image.fromarray(postproc_imgs(results, vae)[0])
    # Optionally draw bbox
    # result_img = draw_bboxes(result_img, entity_bboxes)
    return result_img, debug_mask, seed

#%%
def build_interface(lora_dir: str = "outputs/training"):
    lora_files = sorted([
        str(x).replace(lora_dir+"/", '') 
        for x in Path(lora_dir).glob("*/*.safetensors")
    ])

    with gr.Blocks(title="Whisperer Demo") as demo:
        with gr.Tabs("Single Control-User Input Mode"):
            gr.Markdown("## FramePack Single Image Generation\nEnter a prompt and optional entity box.")
            with gr.Row():
                with gr.Column(scale=2):
                    lora_file = gr.Dropdown(label="LoRA Model", choices=lora_files, value=lora_files[-1], interactive=True)
                    prompt = gr.Textbox(label="Prompt", value="An anime-style girl wearing a school uniform and ribbon tie is walking along the seaside.")
                    bbox = gr.Textbox(label="Entity BBox (x1,y1,x2,y2 normalized or pixels)", value="0.6,0.1,1.0,1.0")
                    with gr.Row():
                        width = gr.Slider(256, 1920, value=480, step=16, label="Width")
                        height = gr.Slider(256, 1920, value=480, step=16, label="Height")
                    control_image = gr.Image(label="Optional Control Image", type="pil")
                    with gr.Accordion("Advanced Settings", open=False):
                        seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                        with gr.Row("Indices"):
                            target_index = gr.Slider(0, 28, value=3, step=1, label="Target Latent Index")
                            control_index = gr.Slider(0, 3, value=0, step=1, label="Control Latent Index")
                        with gr.Row("Control Image Processing"):
                            control_mode = gr.Dropdown(label="Control Adjustment Mode", choices=[
                                "provided_face_bbox",
                                "full_width_full_height", 
                                "full_width_relative_height",
                                "relative_width_full_height",
                                "provided_size_mid_x"
                            ], value="full_width_relative_height", interactive=True)
                            use_facecrop = gr.Checkbox(label="Use Face Cropping for Control", value=True)
                            use_rembg = gr.Checkbox(label="Use Rembg to Remove Control Background", value=False)
                        with gr.Row("Control Image Processing"):
                            scale_c = gr.Slider(0.5, 2.5, value=1.2, step=0.1, label="Control Size Scale (scale_c)")
                            c_width_given = gr.Number(label="Control Width Override (0 to disable)", value=0, precision=0)
                        with gr.Row("Inference Settings"):
                            steps = gr.Slider(5, 50, value=25, step=1, label="Steps")
                            guidance = gr.Slider(1.0, 15.0, value=10.0, step=0.5, label="Distilled Guidance Scale")
                    run_btn = gr.Button("Generate")
                with gr.Column(scale=1):
                    out_image = gr.Image(label="Result", type="pil")
                    out_mask = gr.Image(label="Entity Mask", type="pil")
                    out_seed = gr.Number(label="Used Seed")
                    
            run_btn.click(fn=generate_image, 
                        inputs=[lora_file, prompt, bbox, control_image, width, height, steps, guidance, seed, 
                                target_index, control_index, control_mode, scale_c, c_width_given, use_rembg, use_facecrop], 
                        outputs=[out_image, out_mask, out_seed])
    return demo


def main():
    demo = build_interface()
    demo.launch()

if __name__ == "__main__":
    main()
