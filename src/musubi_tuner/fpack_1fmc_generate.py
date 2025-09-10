import os, sys
sys.path.append("src/")
os.environ["CUDA_VISIBLE_DEVICES"] = "1,"
from types import SimpleNamespace
import math
import numpy as np
from einops import rearrange
from PIL import Image, ImageOps
import torch

from musubi_tuner.dataset.image_video_dataset import resize_image_to_bucket
from musubi_tuner.networks import lora_framepack
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
    image_pil = read_image(image_path).convert("RGB")

    image_np = np.array(image_pil)
    if width is None or height is None:
        target_area = image_pil.size[0]*image_pil.size[1]
        width, height = getres(image_pil.size[0], image_pil.size[1], target_area=target_area, div_factor=16)
    image_np = resize_image_to_bucket(image_np, (width, height))
    image_tensor = (torch.from_numpy(image_np).float() / 127.5 - 1.0).permute(2,0,1)[None, :, None]
    return image_tensor, image_np

def preproc_mask(mask_path, width, height, invert=False):
    if mask_path == '':
        image_pil = Image.new("L", (width // 8, height // 8), 255)
    else:
        image_pil = read_image(mask_path).convert("L")
    if invert:
        image_pil = ImageOps.invert(image_pil)

    image_np = np.array(image_pil)
    if width is not None and height is not None:
        image_np = resize_image_to_bucket(image_np,  (width // 8, height // 8))
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

    def prepare_text_inputs(self, prompt, entity_prompts=[]):
        with torch.autocast(device_type=self.device.type, dtype=self.text_encoder1.dtype), torch.no_grad():
            llama_vecs = []
            llama_vecs_inds = []
            llama_strtokens = {}
            start_ind = 0
            for i, p in enumerate([prompt] + entity_prompts):
                llama_vec_i, clip_l_pooler_i, llama_strtokens_i = encode_prompt_conds(
                    p, self.text_encoder1, self.text_encoder2, self.tokenizer1, self.tokenizer2, 
                    custom_system_prompt=None, return_tokendict=True
                )
                llama_vecs.append(llama_vec_i)
                llama_vecs_inds.append(llama_vec_i.shape[1])
                llama_strtokens.update({start_ind+i:x for i,x in llama_strtokens_i.items()})
                if i == 0:
                    clip_l_pooler = clip_l_pooler_i
                start_ind += llama_vec_i.shape[1]
                
            llama_vec = torch.cat(llama_vecs, dim=1).to(self.device, dtype=self.dtype)
            clip_l_pooler = clip_l_pooler.to(self.device, dtype=self.dtype)
            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            llama_vecs_inds = [(sum(llama_vecs_inds[:i]), sum(llama_vecs_inds[:i+1])) for i,x in enumerate(llama_vecs_inds)]

        llama_vec_n = torch.zeros_like(llama_vec).to(self.device, dtype=self.dtype)
        clip_l_pooler_n  = torch.zeros_like(clip_l_pooler).to(self.device, dtype=self.dtype)
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

    def prepare_image_inputs(self, image_paths, width, height, target_index=[1]):
        image_embeddings, img_nps = [], []
        if type(image_paths) == str:
            image_paths = [image_paths]

        with torch.no_grad():
            for image_path in image_paths:
                _, img_np = preproc_image(image_path, width, height)
                image_encoder_output = hf_clip_vision_encode(img_np, self.feature_extractor, self.image_encoder)
                image_encoder_last_hidden_state = image_encoder_output.last_hidden_state.to(self.device, dtype=self.dtype)
                image_embeddings.append(image_encoder_last_hidden_state)
                img_nps.append(img_np)

        latent_indices = torch.tensor([target_index], dtype=torch.int64)  # 1x1 latent index for target image
        return {
            "image_embeddings" : image_embeddings,
            "latent_indices" : latent_indices,
        } , img_nps

    def prepare_control_inputs(self, control_image_paths, control_image_mask_paths, 
                            width=None, height=None,
                            control_indices=[0,10]):
        control_latents, control_nps = [], []
        for i, (control_image_path, control_mask_path) in enumerate(zip(control_image_paths, control_image_mask_paths)):
            c_img_tensor, c_img_np = preproc_image(control_image_path, width, height)
            c_H, c_W = c_img_tensor.shape[-2:]
            c_img_latent = vae_encode(c_img_tensor, self.vae).cpu()
            c_mask_image, c_mask_np = preproc_mask(control_mask_path, c_W, c_H)
            c_img_latent = c_img_latent * c_mask_image
            control_latents.append(c_img_latent)
            control_nps.append(np.concatenate([c_img_np, resize_image_to_bucket(c_mask_np, (c_W, c_H))[..., None]], -1))

        clean_latents = control_latents
        clean_latent_indices = [torch.tensor([[ind]], dtype=torch.int64) for ind in control_indices]

        return {
            "clean_latents" : clean_latents, 
            "clean_latent_indices" : clean_latent_indices,
            "clean_latents_2x" : None, 
            "clean_latent_2x_indices" : None,
            "clean_latents_4x" : None, 
            "clean_latent_4x_indices" : None,
        } , control_nps

    # def prepare_control_inputs(self, control_image_paths, control_image_mask_paths, width, height, control_indices=[0, 10]):
    #     control_latents, control_nps = [], []
    #     for i, (control_image_path, control_mask_path) in enumerate(zip(control_image_paths, control_image_mask_paths)):
    #         c_img_tensor, c_img_np = preproc_image(control_image_path, width, height)
    #         c_img_latent = vae_encode(c_img_tensor, self.vae).cpu()
            
    #         c_mask_image, c_mask_np = preproc_mask(control_mask_path, width, height)
    #         c_img_latent = c_img_latent * c_mask_image
    #         control_latents.append(c_img_latent)
    #         control_nps.append(np.concatenate([c_img_np, resize_image_to_bucket(c_mask_np, (width, height))[..., None]], -1))
    #     clean_latents = torch.cat(control_latents, dim=2)  # (1, 16, num_control_images, H//8, W//8)

    #     clean_latent_indices = torch.tensor([control_indices], dtype=torch.int64)
    #     return clean_latents, clean_latent_indices

    def __call__(self, 
            prompt, 
            image_paths, 
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
            entity_prompts = [],
            cache_results = False
        ):
        if height is None or width is None:
            og_width, og_height = read_image(image_paths[0]).size
            width, height = getres(og_width, og_height)

        text_kwargs = self.prepare_text_inputs(prompt)

        image_kwargs, img_np = self.prepare_image_inputs(
            image_paths, 
            width, height, 
            target_index=target_index
        )
        # clean_latents, clean_latent_indices = self.prepare_control_inputs(
        #     control_image_paths, 
        #     control_image_mask_paths, 
        #     width, height, 
        #     control_indices=control_indices
        # )
        control_kwargs, control_nps = self.prepare_control_inputs(
            control_image_paths, control_image_mask_paths,
            width=width, height=height,
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
            generator=generator, device=self.device, dtype=self.dtype,
            cache_results=cache_results,
            **text_kwargs,
            **image_kwargs,
            **control_kwargs
        )
    
        history_pixels = torch.cat([
            vae_decode(results[:, :, i:i+1, :, :], self.vae).cpu() for i in range(results.shape[2])
        ], dim=2)
        result_img = Image.fromarray(to_img(history_pixels[0,:,0,:,:]))

        if cache_results:
            return result_img, attn_cache
        else:
            return result_img