from pathlib import Path, PosixPath
from typing import Tuple, List
import math
import numpy as np
import json
from omegaconf import OmegaConf
from PIL import Image, ImageOps, ImageDraw
# from rembg import remove, new_session
import torch
from transformers import pipeline

from musubi_tuner.dataset.image_video_dataset import resize_image_to_bucket
from musubi_tuner.frame_pack.clip_vision import hf_clip_vision_encode
from musubi_tuner.frame_pack.hunyuan import encode_prompt_conds, vae_encode, vae_decode
from musubi_tuner.frame_pack.utils import crop_or_pad_yield_mask
from musubi_tuner.utils.bbox_utils import get_bbox_from_mask, get_mask_from_bboxes, get_facebbox_from_bbox, auto_scale_layout_data
from musubi_tuner.utils.bbox_utils import draw_bboxes, draw_bboxes_images
from musubi_tuner.utils.keypalign_utils import search_facebbox_for_layout

# rembg_session = new_session('u2net', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
rmbg14_session = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

def getres(orig_width, orig_height, target_area=1024*1024, div_factor=16, max_aspect_ratio=None):
    if target_area is not None:
        aspect_ratio = orig_width / orig_height
        
        # Constrain aspect ratio if max_aspect_ratio is specified
        if max_aspect_ratio is not None:
            if aspect_ratio > max_aspect_ratio:
                aspect_ratio = max_aspect_ratio
            elif aspect_ratio < 1 / max_aspect_ratio:
                aspect_ratio = 1 / max_aspect_ratio
        
        new_height = math.sqrt(target_area / aspect_ratio)
        new_width = target_area / new_height
    else:
        new_width, new_height = orig_width, orig_height
    new_width = int(round(new_width / div_factor) * div_factor)
    new_height = int(round(new_height / div_factor) * div_factor)

    return new_width, new_height

SDXL_RESOLUTIONS: List[Tuple[int, int]] = [
    # Square
    (768, 768), (1024, 1024), (1152, 1152),

    # Portrait (4:5, 2:3, 3:4)
    (832, 1040), (832, 1216), (896, 1152), (960, 1280),

    # Landscape (5:4, 3:2, 4:3, etc.)
    (1040, 832), (1216, 832), (1152, 896), (1280, 960), (1344, 768),

    # True 16:9 / 9:16
    (1152, 648), (1280, 720), (1536, 864),
    (896, 1600), (1024, 1792),

    # Ultra-wide & tall
    (1536, 640), (1600, 640), (640, 1536),
]

def _aspect_ratio(w: int, h: int) -> float:
    return w / h

def _clamp_ratio(r: float, max_ratio: float = 2.5) -> float:
    """Clamp aspect ratio into [1/max_ratio, max_ratio]."""
    lo = 1.0 / max_ratio
    if r < lo:
        return lo
    if r > max_ratio:
        return max_ratio
    return r

def pick_sdxl_resolution(
    init_size: Tuple[int, int],
    candidates: List[Tuple[int, int]] = SDXL_RESOLUTIONS,
    tie_pref_megapixels: float = 1.0  # prefer ~1MP if aspect ties
) -> Tuple[int, int]:
    """
    Given (width, height), return the closest SDXL-friendly resolution
    from `candidates`, preserving aspect as much as possible.
    If the aspect ratio exceeds 2.5 (or is below 1/2.5), it's clamped first.
    """
    w, h = init_size
    if w <= 0 or h <= 0:
        raise ValueError("Width and height must be positive.")

    target_ar = _clamp_ratio(_aspect_ratio(w, h), max_ratio=2.5)

    # Find candidate with minimal |aspect - target_ar|.
    # Tie-break by closeness of area (in MP) to `tie_pref_megapixels`.
    best = None
    best_key = None
    for cw, ch in candidates:
        cand_ar = _aspect_ratio(cw, ch)
        ar_diff = abs(cand_ar - target_ar)
        area_mp = (cw * ch) / 1_000_000.0
        # Sort key: primary by aspect diff, secondary by |area - preferred|
        key = (ar_diff, abs(area_mp - tie_pref_megapixels))
        if best_key is None or key < best_key:
            best_key = key
            best = (cw, ch)
    return best

def get_panel_layout(layout, page_w, page_h):
    width_orig, height_orig = (layout['bbox'][2]-layout['bbox'][0])/1000*page_w, (layout['bbox'][3]-layout['bbox'][1])/1000*page_h
    width, height = pick_sdxl_resolution((width_orig, height_orig))
    panel_layout = {i: {
        'bbox': list(map(lambda x: x/1000, x[:4])), 
        'body': list(map(lambda x: x/1000, x[4:]))
    } for i,x in enumerate(layout['body'])}
    return panel_layout, width, height

#%%
def get_text_preproc(prompt, text_encoder1, text_encoder2, tokenizer1, tokenizer2, entity_prompts=[], device=torch.device('cuda'), dtype=torch.bfloat16):
    with torch.autocast(device_type=device.type, dtype=text_encoder1.dtype), torch.no_grad():
        llama_vecs = []
        llama_vecs_inds = []
        llama_strtokens = {}
        start_ind = 0
        for i, p in enumerate([prompt] + entity_prompts):
            llama_vec_i, clip_l_pooler_i, llama_strtokens_i = encode_prompt_conds(
                p, text_encoder1, text_encoder2, tokenizer1, tokenizer2, custom_system_prompt=None, return_tokendict=True
            )
            llama_vecs.append(llama_vec_i)
            llama_vecs_inds.append(llama_vec_i.shape[1])
            llama_strtokens.update({start_ind+i:x for i,x in llama_strtokens_i.items()})
            if i == 0:
                clip_l_pooler = clip_l_pooler_i
            start_ind += llama_vec_i.shape[1]
            
        llama_vec = torch.cat(llama_vecs, dim=1).to(device, dtype=dtype)
        clip_l_pooler = clip_l_pooler.to(device, dtype=dtype)
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vecs_inds = [(sum(llama_vecs_inds[:i]), sum(llama_vecs_inds[:i+1])) for i,x in enumerate(llama_vecs_inds)]

    llama_vec_n = torch.zeros_like(llama_vec).to(device, dtype=dtype)
    clip_l_pooler_n  = torch.zeros_like(clip_l_pooler).to(device, dtype=dtype)
    llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
    return {
        "prompt_embeds" : llama_vec,
        "prompt_embeds_mask" : llama_attention_mask,
        "prompt_poolers" : clip_l_pooler,
        "prompt_entity_inds" : llama_vecs_inds,
        "prompt_strtokens" : llama_strtokens,
        "negative_prompt_embeds" : llama_vec_n,
        "negative_prompt_embeds_mask" : llama_attention_mask_n,
        "negative_prompt_poolers" : clip_l_pooler_n,
    }

def getres(orig_width, orig_height, target_area=480*480, div_factor=16, max_aspect_ratio=None):
    if target_area is not None:
        aspect_ratio = orig_width / orig_height
        
        # Constrain aspect ratio if max_aspect_ratio is specified
        if max_aspect_ratio is not None:
            if aspect_ratio > max_aspect_ratio:
                aspect_ratio = max_aspect_ratio
            elif aspect_ratio < 1 / max_aspect_ratio:
                aspect_ratio = 1 / max_aspect_ratio
        
        new_height = math.sqrt(target_area / aspect_ratio)
        new_width = target_area / new_height
    else:
        new_width, new_height = orig_width, orig_height
    new_width = int(round(new_width / div_factor) * div_factor)
    new_height = int(round(new_height / div_factor) * div_factor)

    return new_width, new_height

def preproc_image(image_path, width=None, height=None, use_rembg=False):
    if type(image_path) in [str, Path, PosixPath]:
        image_pil = Image.open(image_path).convert("RGB")
    elif type(image_path) == np.ndarray:
        image_pil = Image.fromarray(image_path)
    else:
        image_pil = image_path
    if use_rembg:
        image_pil = rmbg14_session(image_pil)
        new_image = Image.new("RGBA", image_pil.size, "WHITE")
        new_image.paste(image_pil, (0, 0), image_pil)
        image_pil = new_image.convert('RGB')
    image_np = np.array(image_pil)

    if width is None and height is None:
        target_area = image_pil.size[0]*image_pil.size[1]
        width, height = getres(image_pil.size[0], image_pil.size[1], target_area=target_area, div_factor=16)
    elif width is not None and height is None:
        width = (width // 16) * 16
        height = int(round(width * image_pil.size[1] / image_pil.size[0]) // 2 * 2)
    elif height is not None and width is None:
        height = (height // 16) * 16
        width = int(round(height * image_pil.size[0] / image_pil.size[1]) // 2 * 2)
    else:
        width, height = (width // 16) * 16, (height // 16) * 16

    image_np = resize_image_to_bucket(image_np, (width, height))
    image_tensor = (torch.from_numpy(image_np).float() / 127.5 - 1.0).permute(2,0,1)[None, :, None]
    return image_tensor, image_np

def preproc_mask(mask_path, width=None, height=None, invert=False):
    if type(mask_path) in [str, Path, PosixPath]:
        if mask_path == '':
            image_pil = Image.new("L", (width // 8, height // 8), 255)
        else:
            image_pil = Image.open(mask_path).convert("L")
    elif type(mask_path) == np.ndarray:
        image_pil = Image.fromarray(mask_path)
    else:
        image_pil = mask_path
    
    if invert:
        image_pil = ImageOps.invert(image_pil)
        
    image_np = np.array(image_pil)
    if width is not None and height is not None:
        image_np = resize_image_to_bucket(image_np,  (width // 8, height // 8))
    image_tensor = (torch.from_numpy(image_np).float() / 255.0)[None, None, None, :, :]
    return image_tensor, image_np

def prepare_image_inputs(image_paths, feature_extractor, image_encoder, width=None, height=None, 
                         target_index=1, device=torch.device('cuda'), dtype=torch.bfloat16):
    image_embeddings, img_nps = [], []
    if type(image_paths) == str:
        image_paths = [image_paths]

    with torch.no_grad():
        for image_path in image_paths:
            _, img_np = preproc_image(image_path, width, height)
            image_encoder_output = hf_clip_vision_encode(img_np, feature_extractor, image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state.to(device, dtype=dtype)
            image_embeddings.append(image_encoder_last_hidden_state)
            img_nps.append(img_np)

    latent_indices = torch.tensor([target_index], dtype=torch.int64)  # 1x1 latent index for target image
    return {
        "image_embeddings" : image_embeddings,
        "latent_indices" : latent_indices,
    } , img_nps

def prepare_control_inputs(control_image_paths, control_image_mask_paths, control_image_sizes, 
                           vae, 
                           control_indices=[0,10]):
    if control_image_sizes == None:
        control_image_sizes = [(None, None) for i in range(len(control_image_paths))]

    control_latents, control_nps = [], []
    for i, (control_image_path, control_mask_path) in enumerate(zip(control_image_paths, control_image_mask_paths)):
        c_img_tensor, c_img_np = preproc_image(control_image_path, control_image_sizes[i][0], control_image_sizes[i][1])
        c_H, c_W = c_img_tensor.shape[-2:]
        c_img_latent = vae_encode(c_img_tensor, vae).cpu()
        c_mask_image, c_mask_np = preproc_mask(control_mask_path, c_W, c_H)
        c_img_latent = c_img_latent * c_mask_image
        control_latents.append(c_img_latent)
        control_nps.append(np.concatenate([c_img_np, resize_image_to_bucket(c_mask_np, (c_W, c_H))[..., None]], -1))
    # clean_latents = torch.cat(control_latents, dim=2)  # (1, 16, num_control_images, H//8, W//8)
    clean_latents = control_latents
    clean_latent_indices = [torch.tensor([[ind]], dtype=torch.int64) for ind in control_indices]
    # clean_latent_indices = torch.tensor([control_indices], dtype=torch.int64)

    return {
        "clean_latents" : clean_latents, 
        "clean_latent_indices" : clean_latent_indices,
        "clean_latents_2x" : None, 
        "clean_latent_2x_indices" : None,
        "clean_latents_4x" : None, 
        "clean_latent_4x_indices" : None,
    } , control_nps

def prepare_control_inputs_for_entity(control_image_paths, entity_bboxes, width, height, vae, 
                                      control_image_sizes, face_entity_bboxes=None,
                                      control_indices=[0], latent_indices=[3], 
                                      adjust_custom_wh=False, mode="provided_face_bbox", 
                                      use_rembg=True):
    
    control_latents, control_nps, clean_latent_bboxes = [], [], []
    if len(control_image_paths) == 0:
        return {
            'latent_indices': torch.tensor([latent_indices], dtype=torch.int64),
            'clean_latent_bboxes': [],
            'clean_latents_2x': None,
            'clean_latent_2x_indices': None,
            'clean_latents_4x': None,
            'clean_latent_4x_indices': None
        }, []
    
    for i, control_image_path in enumerate(control_image_paths):
        bbox = entity_bboxes[i]
        c_width, c_height = control_image_sizes[i] if control_image_sizes is not None else (None, None)
        if adjust_custom_wh:
            if c_width is not None:
                c_width, c_height = min(int((c_width * 1.2) // 16 * 16), int(width * (bbox[2]-bbox[0]))), None
            else:
                c_width, c_height = int(width * (bbox[2]-bbox[0])), None

        c_img_tensor, c_img_np = preproc_image(control_image_path, c_width, c_height, use_rembg=use_rembg)
        c_width, c_height = c_img_tensor.shape[4], c_img_tensor.shape[3]
        
        face_bbox = face_entity_bboxes[i] if face_entity_bboxes is not None else None
        if face_bbox is None:
            face_bbox = get_facebbox_from_bbox(bbox, c_width, c_height, width, height, face_bbox=face_bbox, mode=mode)
        clean_latent_bboxes.append(face_bbox)
        c_img_latent = vae_encode(c_img_tensor, vae).cpu()
        control_latents.append(c_img_latent)
        control_nps.append(c_img_np)

    clean_latents = control_latents
    clean_latent_indices = torch.tensor([control_indices], dtype=torch.int64)
    latent_indices = torch.tensor([latent_indices], dtype=torch.int64)
    clean_latent_bboxes = torch.tensor([clean_latent_bboxes], dtype=torch.float32)
    return {
        "latent_indices" : latent_indices,
        "clean_latents" : clean_latents, 
        "clean_latent_indices" : clean_latent_indices,
        'clean_latent_bboxes': clean_latent_bboxes,
        "clean_latents_2x" : None, 
        "clean_latent_2x_indices" : None,
        "clean_latents_4x" : None, 
        "clean_latent_4x_indices" : None,
    } , control_nps

def get_all_kwargs_from_opens2v_metapath(metapath, steps=25, seed=-1,
    width=None, height=None, frames=1, batch_size=1,
    cache_results=True, cache_layers=[], 
    entities = [0],
    device='cuda:0'
    ):
    meta = OmegaConf.load(metapath / f"meta.yaml")

    prompt = meta['cap'][0]
    height = meta['height'] if height is None else height
    width = meta['width'] if width is None else width

    entity_mask_paths = [metapath / f"target_bodmask_{i}.png" for i in entities]
    control_image_paths = [metapath / f"source_facecrop_{i}.png" for i in entities]
    gt_image_path = metapath / f"target_frame.png"

    num_inference_steps=steps
    seed = np.random.randint(2**31) if seed == -1 else seed
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    general_kwargs = {
        'prompt': prompt,
        'sampler': 'unipc',
        'width': width,
        'height': height,
        'frames': frames,
        'batch_size': batch_size,
        'real_guidance_scale': 1.0,
        'distilled_guidance_scale': 10.0,
        'guidance_rescale': 0.0,
        'shift': None,
        'num_inference_step': num_inference_steps,
    }
    rand_kwargs = {
        'generator': generator,
        'device': device,
        'dtype': torch.bfloat16
    }
    cache_kwargs = {
        'cache_results' : cache_results,
        'cache_layers': cache_layers
    }

    # entity_masks = [get_mask_from_bboxes(entity_bboxes, width, height)]
    entity_masks = torch.cat([
        preproc_mask(e_mask, width, height, invert=False)[0] 
        for e_mask in entity_mask_paths
    ], 1)
    entity_masks_np = []
    bboxes = []
    for i in range(entity_masks.shape[0]):
        mask = entity_masks[i,:,0,:,:].permute(1,2,0).cpu().numpy().astype(bool)[...,0]
        entity_masks_np.append(mask)
        bboxes.append(get_bbox_from_mask(mask))

    attn_kwargs = {
        'use_attention_masking': ["mask_control"],
        'entity_masks': entity_masks,
        "control_image_paths": control_image_paths,
    }
    return {
        **general_kwargs,
        **rand_kwargs,
        **cache_kwargs, 
        **attn_kwargs,
    }, {
        'entitymask_nps': entity_masks_np,
        'bboxes' : bboxes,
        'gt_np': np.array(Image.open(gt_image_path).convert("RGB").resize((width, height))),
    }

def get_info_from_vistorybench(dataset, story_num, shot_num):
    story = dataset.load_story(story_num)
    characters = dataset.load_characters(story_num)
    
    story_dict = {x['index']:x for x in story['shots']}
    story_shot = story_dict[shot_num]
    story_shot['type'] = story['type']
    characters_shot = {k: characters[k] for k in story_shot['character_name']}
    characters_tags = set([v['tag'] for k,v in characters.items()])
    if 'non_human' in characters_tags or 'unrealistic_human' in characters_tags:
        story_shot['type'] = 'Illustration, ' + story_shot['type']
    else:
        story_shot['type'] = 'Realistic, ' + story_shot['type']

    char_prompts = []
    for char_key, char_name in zip(story_shot['character_key'], story_shot['character_name']):
        char_prompt = characters[char_key]['prompt']
        char_prompts.append(f'{char_name} is {char_prompt}')

    # prompt = story_shot['type'] + ". " + ", ".join(story_shot['scene'].split(", ")[:max_scene_sentences]) + ". " + story_shot['script']
    prompt = (
        f"{story_shot['type']};"
        f"{story_shot['camera']};"
        f"{story_shot['script']};"
        f"{story_shot['scene']};"
        f"{story_shot['plot']};" 
        f"{';'.join(char_prompts)};" # Separate multiple character descriptions with ;
    )

    return story_shot, characters_shot, prompt

def parse_bodylayout(layout_dir):
    layout = json.loads(Path(layout_dir).read_text())
    body_layout_dict = {}
    for key, panel_layout in layout.items():
        body_layout = {
            i: {
                'bbox' : list(map(lambda a: a/1000, x[:4])), 
                'body' : np.reshape(x[4:], (-1,2)) / 1000
            } for i, x in enumerate(panel_layout['body'])
        }
        body_layout_dict[key] = (panel_layout['bbox'], body_layout)
    return body_layout_dict

def postproc_imgs(results, vae):
    pixels = torch.cat([
        vae_decode(results[:, :, i:i + 1, :, :], vae).cpu() 
        for i in range(results.shape[2])
    ], dim=2)
        
    result_imgs = []
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[2]):
            pixel = (pixels[i,:,j,:,:]+1.0)/2.0
            pixel = torch.clamp(pixel.permute(1,2,0).cpu(), 0.0, 1.0)
            result_img = (pixel * 255.).numpy().astype(np.uint8)
            result_alpha =  np.ones(result_img.shape[:2] + (1, ))*255
            result_img = np.concatenate([result_img, result_alpha], axis=-1).astype(np.uint8)
            result_imgs.append(result_img)
    return result_imgs  

def get_all_control_kwargs(panel_layout, characters_shot, vae,
    width=1344, height=768, 
    crop_face_detect=True, use_face_detect=True, 
    c_width_given=None, scale_c=1.2, use_safety=True,
    bbox_mode="relative_width_full_height",
    control_indices=[0], latent_indices=[3], use_rembg=True,
    use_auto_scale=False, max_chara_imgs=3, max_chara=4
    ):

    auto_scaled_layout = panel_layout
    if use_auto_scale:
        auto_scaled_layout, metadata = auto_scale_layout_data(panel_layout)
        
    debug_dict = search_facebbox_for_layout(
        auto_scaled_layout, characters_shot, (width, height), 
        crop_face_detect=crop_face_detect, use_face_detect=use_face_detect,
        c_width_given=c_width_given, scale_c=scale_c, use_safety=use_safety,
        bbox_mode=bbox_mode, max_chara_imgs=max_chara_imgs)

    print_res = ""
    for k,v in debug_dict.items():
        print_res += f"Entity {k} (Use Crop: {True})\n"
        print_res += f"\tControl Image Path: {v['control_image_path']}\n"
        print_res += f"\tControl Image Size: {v['control_image_size']}\n"
        print_res += "\tAttn BBox: [" + ', '.join([f"{b:.3f}" for b in v['entity_bbox']]) + "]\n"
        print_res += "\tFace BBox: [" + ', '.join([f"{b:.3f}" for b in v['face_bbox']]) + "]\n"

    n_chara = min(len(debug_dict), max_chara)
    if len(debug_dict) == 0:
        control_images = []
        control_image_sizes = []
        entity_bboxes = []
        face_bboxes = []
        entity_masks = None
        debug_mask = Image.new("RGB", (width, height), (0,0,0))
    else:
        control_images = [debug_dict[k]['control_image'] for k in list(debug_dict.keys())[:n_chara]]
        control_image_sizes = [debug_dict[k]['control_image_size'] for k in list(debug_dict.keys())[:n_chara]]
        entity_bboxes = [debug_dict[k]['entity_bbox'] for k in list(debug_dict.keys())[:n_chara]]
        face_bboxes = [debug_dict[k]['face_bbox'] for k in list(debug_dict.keys())[:n_chara]]
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

    return control_kwargs, entity_masks, control_nps, debug_mask, print_res
