from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageOps
import torch

from musubi_tuner.dataset.image_video_dataset import resize_image_to_bucket
from musubi_tuner.frame_pack.clip_vision import hf_clip_vision_encode
from musubi_tuner.frame_pack.hunyuan import encode_prompt_conds, vae_encode
from musubi_tuner.frame_pack.utils import crop_or_pad_yield_mask


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

def getres(orig_width, orig_height, target_area=480*480, div_factor=16):
    if target_area is not None:
        aspect_ratio = orig_width / orig_height
        new_height = math.sqrt(target_area / aspect_ratio)
        new_width = target_area / new_height
    else:
        new_width, new_height = orig_width, orig_height
    new_width = int(round(new_width / div_factor) * div_factor)
    new_height = int(round(new_height / div_factor) * div_factor)

    return new_width, new_height

def preproc_image(image_path, width=None, height=None):
    if type(image_path) in [str, Path]:
        image_pil = Image.open(image_path).convert("RGB")
    elif type(image_path) == np.ndarray:
        image_pil = Image.fromarray(image_path)
    else:
        image_pil = image_path
    image_np = np.array(image_pil)

    if width is None or height is None:
        target_area = image_pil.size[0]*image_pil.size[1]
        width, height = getres(image_pil.size[0], image_pil.size[1], target_area=target_area, div_factor=16)
    image_np = resize_image_to_bucket(image_np, (width, height))
    image_tensor = (torch.from_numpy(image_np).float() / 127.5 - 1.0).permute(2,0,1)[None, :, None]
    return image_tensor, image_np

def preproc_mask(mask_path, width=None, height=None, invert=False):
    if mask_path == '':
        image_pil = Image.new("L", (width // 8, height // 8), 255)
    else:
        image_pil = Image.open(mask_path).convert("L")
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

def prepare_control_inputs(control_image_paths, control_image_mask_paths, vae, 
                           width=None, height=None,
                           control_indices=[0,10]):
    control_latents, control_nps = [], []
    for i, (control_image_path, control_mask_path) in enumerate(zip(control_image_paths, control_image_mask_paths)):
        c_img_tensor, c_img_np = preproc_image(control_image_path, width, height)
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