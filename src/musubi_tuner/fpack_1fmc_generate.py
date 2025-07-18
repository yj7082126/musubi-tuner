import os, sys
sys.path.append("src/")
os.environ["CUDA_VISIBLE_DEVICES"] = "1,"
from types import SimpleNamespace
import math
import numpy as np
from einops import rearrange
from PIL import Image, ImageDraw
import torch

from musubi_tuner.dataset.image_video_dataset import resize_image_to_bucket
from musubi_tuner.networks import lora_framepack
from musubi_tuner.frame_pack.uni_pc_fm import sample_unipc
from musubi_tuner.frame_pack.wrapper import fm_wrapper
from musubi_tuner.frame_pack.clip_vision import hf_clip_vision_encode
from musubi_tuner.frame_pack.framepack_utils import load_vae, load_text_encoder1, load_text_encoder2, load_image_encoders
from musubi_tuner.frame_pack.hunyuan import encode_prompt_conds, vae_encode, vae_decode
from musubi_tuner.frame_pack.hunyuan_video_packed import load_packed_model, attn_cache
from musubi_tuner.frame_pack.k_diffusion_hunyuan import sample_hunyuan
from musubi_tuner.frame_pack.utils import crop_or_pad_yield_mask
from musubi_tuner.wan_generate_video import merge_lora_weights

def read_image(image_path):
    if type(image_path) == Image.Image:
        image_pil = image_path
    elif type(image_path) == np.ndarray:
        image_pil = Image.fromarray(image_path)
    else:
        image_pil = Image.open(image_path)
    return image_pil

def getres(orig_width, orig_height, target_area=480*480, div_factor=16):
    aspect_ratio = orig_width / orig_height

    new_height = math.sqrt(target_area / aspect_ratio)
    new_width = target_area / new_height

    new_width = int(round(new_width / div_factor) * div_factor)
    new_height = int(round(new_height / div_factor) * div_factor)

    return new_width, new_height

def preproc_image(image_path, width, height):
    image_pil = read_image(image_path).convert("RGB")
    image_np = resize_image_to_bucket(np.array(image_pil), (width, height))
    image_tensor = (torch.from_numpy(image_np).float() / 127.5 - 1.0).permute(2,0,1)[None, :, None]
    return image_tensor, image_np

def preproc_mask(mask_path, width, height):
    if mask_path == '':
        image_pil = Image.new("L", (width // 8, height // 8), 255)
    else:
        image_pil = read_image(mask_path).convert("L")
    image_np = resize_image_to_bucket(np.array(image_pil), (width // 8, height // 8))
    image_tensor = (torch.from_numpy(image_np).float() / 255.0)[None, None, None, :, :]
    return image_tensor, image_np

def to_img(x):
    x = torch.clamp((x+1.0)/2.0, 0.0, 1.0)
    x = x.permute(1,2,0).cpu().numpy()
    x = (x * 255.).astype(np.uint8)
    return x

class FramePack_1fmc():
    def __init__(self,
        dit_path = "/data/stale/patrickkwon/video/stable-diffusion-webui/models/Hunyuan/FramePackI2V_HY_bf16.safetensors",
        vae_path = "/data/stale/patrickkwon/video/stable-diffusion-webui/models/VAE/hunyuan-video-t2v-720p-vae.pt",
        text_encoder1_path = "/shared/video/ComfyUI/models/text_encoders/llava_llama3_fp16.safetensors",
        text_encoder2_path = "/shared/video/ComfyUI/models/text_encoders/clip_l.safetensors",
        image_encoder_path = "/shared/video/ComfyUI/models/clip_vision/sigclip_vision_patch14_384.safetensors",
        lora_path = "/data/stale/patrickkwon/video/stable-diffusion-webui/models/Lora/framepack/fpack_1fmc_bg_lora/bg_lora_1000.safetensors",
        lora_multiplier = 1.0,
        device = torch.device('cuda:0'), 
        dtype = torch.bfloat16
    ):
        self.device = device
        self.dtype = dtype

        self.model = load_packed_model(self.device, dit_path, 'sageattn', self.device)
        self.model.to(self.device)
        self.model.eval().requires_grad_(False)
        # model.move_to_device_except_swap_blocks(device)
        # model.prepare_block_swap_before_forward()
        if lora_path is not None:
            merge_lora_weights(
                lora_framepack, 
                self.model, 
                SimpleNamespace(
                    lora_weight = [lora_path], 
                    lora_multiplier = [lora_multiplier], 
                    include_patterns=None, 
                    exclude_patterns=None, 
                    lycoris=None,
                    save_merged_model=False), 
                device, None
            )

        self.vae = load_vae(vae_path, 32, 128, device)

        self.tokenizer1, self.text_encoder1 = load_text_encoder1(SimpleNamespace(text_encoder1=text_encoder1_path), False, device)
        self.tokenizer2, self.text_encoder2 = load_text_encoder2(SimpleNamespace(text_encoder2=text_encoder2_path))
        self.feature_extractor, self.image_encoder = load_image_encoders(SimpleNamespace(image_encoder=image_encoder_path))

    def prepare_text_inputs(self, prompt):
        with torch.autocast(device_type=self.device.type, dtype=self.text_encoder1.dtype), torch.no_grad():
            llama_vec, clip_l_pooler = encode_prompt_conds(
                prompt, self.text_encoder1, self.text_encoder2, self.tokenizer1, self.tokenizer2, custom_system_prompt=None
            )
            llama_vec = llama_vec.to(self.device, dtype=self.dtype)
            clip_l_pooler = clip_l_pooler.to(self.device, dtype=self.dtype)
            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        return llama_vec, clip_l_pooler, llama_attention_mask

    def prepare_negative_text_inputs(self, 
                                     llama_vec_shape=[1,512,4096], 
                                     clip_l_shape=[1, 768]):
        llama_vec_n = torch.zeros(llama_vec_shape).to(self.device, dtype=self.dtype)
        clip_l_pooler_n = torch.zeros(clip_l_shape).to(self.device, dtype=self.dtype)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        return llama_vec_n, clip_l_pooler_n, llama_attention_mask_n

    def prepare_image_inputs(self, image_path, width, height, target_index=[1]):
        img_tensor, img_np = preproc_image(image_path, width, height)
        with torch.no_grad():
            image_encoder_output = hf_clip_vision_encode(img_np, self.feature_extractor, self.image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state.to(self.device, dtype=self.dtype)

        latent_indices = torch.tensor([target_index], dtype=torch.int64)  # 1x1 latent index for target image
        return image_encoder_last_hidden_state, latent_indices

    def prepare_control_inputs(self, control_image_paths, control_image_mask_paths, width, height, control_indices=[0, 10]):
        control_latents, control_nps = [], []
        for i, (control_image_path, control_mask_path) in enumerate(zip(control_image_paths, control_image_mask_paths)):
            c_img_tensor, c_img_np = preproc_image(control_image_path, width, height)
            c_img_latent = vae_encode(c_img_tensor, self.vae).cpu()
            
            c_mask_image, c_mask_np = preproc_mask(control_mask_path, width, height)
            c_img_latent = c_img_latent * c_mask_image
            control_latents.append(c_img_latent)
            control_nps.append(np.concatenate([c_img_np, resize_image_to_bucket(c_mask_np, (width, height))[..., None]], -1))
        clean_latents = torch.cat(control_latents, dim=2)  # (1, 16, num_control_images, H//8, W//8)

        clean_latent_indices = torch.tensor([control_indices], dtype=torch.int64)
        return clean_latents, clean_latent_indices

    def __call__(self, 
            prompt, 
            image_path, 
            control_image_paths, 
            control_image_mask_paths, 
            height = None, 
            width = None, 
            num_inference_steps = 25,
            real_guidance_scale = 1.0,
            distilled_guidance_scale = 10.0,
            guidance_rescale = 0.0,
            target_index = [1], 
            control_indices = [0,10], 
            seed = 48, 
            cache_results = False
        ):
        
        if height is None or width is None:
            og_width, og_height = read_image(image_path).size
            width, height = getres(og_width, og_height)

        prompt_embeds, prompt_poolers, prompt_embeds_mask = self.prepare_text_inputs(prompt)
        negative_prompt_embeds, negative_prompt_poolers, negative_prompt_embeds_mask = self.prepare_negative_text_inputs(
            prompt_embeds.shape, prompt_poolers.shape
        )

        image_embeddings, latent_indices = self.prepare_image_inputs(
            image_path, 
            width, height, 
            target_index=target_index
        )
        clean_latents, clean_latent_indices = self.prepare_control_inputs(
            control_image_paths, 
            control_image_mask_paths, 
            width, height, 
            control_indices=control_indices
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        results = sample_hunyuan(
            transformer=self.model,
            sampler='unipc',
            width=width,
            height=height,
            frames=1,
            real_guidance_scale=real_guidance_scale,
            distilled_guidance_scale=distilled_guidance_scale,
            guidance_rescale=guidance_rescale,
            shift=None,
            num_inference_steps=num_inference_steps,
            generator=generator,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            prompt_poolers=prompt_poolers,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            negative_prompt_poolers=negative_prompt_poolers,
            device=self.device,
            dtype=self.dtype,
            image_embeddings=image_embeddings,
            latent_indices=latent_indices,
            clean_latents=clean_latents,
            clean_latent_indices=clean_latent_indices,
            clean_latents_2x=None,
            clean_latent_2x_indices=None,
            clean_latents_4x=None,
            clean_latent_4x_indices=None,
            cache_results=cache_results
        )
    
        history_pixels = torch.cat([
            vae_decode(results[:, :, i:i+1, :, :], self.vae).cpu() for i in range(results.shape[2])
        ], dim=2)
        result_img = Image.fromarray(to_img(history_pixels[0,:,0,:,:]))

        if cache_results:
            return result_img, attn_cache
        else:
            return result_img