from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

from .networks import lora_framepack
from .frame_pack.framepack_utils import load_vae, load_text_encoder1, load_text_encoder2
from .frame_pack.hunyuan_video_packed import load_packed_model, attn_cache
from .frame_pack.k_diffusion_hunyuan import sample_hunyuan
from .wan_generate_video import merge_lora_weights
from .utils.attn_utils import get_pltplot_as_pil
from .utils.preproc_utils import postproc_imgs, get_text_preproc, get_all_control_kwargs
from .utils.viz_utils import printable_metadata, return_total_visualization

class FramePack_1fmc():
    def __init__(self,
        dit_path = "/lustre/fs1/home/yo564250/workspace/ComfyUI/models/diffusion_models/FramePackI2V_HY_bf16.safetensors",
        vae_path = "/lustre/fs1/home/yo564250/workspace/ComfyUI/models/vae/hunyuan-video-t2v-720p-vae.pt",
        text_encoder1_path = "/lustre/fs1/home/yo564250/workspace/ComfyUI/models/text_encoders/llava_llama3_fp16.safetensors",
        text_encoder2_path = "/lustre/fs1/home/yo564250/workspace/ComfyUI/models/text_encoders/clip_l.safetensors",
        lora_path = "../../outputs/training/idmask_control_lora_wrope_v2/idmask_control_lora_wrope_v2_5-step00004000.safetensors",
        lora_multiplier = 1.0,
        device = torch.device('cuda:0'), 
        dtype = torch.bfloat16
    ):
        self.device = device
        self.dtype = dtype

        self.model = load_packed_model(self.device, dit_path, 'sageattn', self.device, has_image_proj=False)
        self.model.to(self.device)
        self.model.eval().requires_grad_(False)

        self.lora_path = lora_path
        if self.lora_path is not None:
            merge_lora_weights(
                lora_framepack, 
                self.model, 
                SimpleNamespace(
                    lora_weight = [self.lora_path], 
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
        print("Model, VAE, Text Encoders, LoRA loaded.")

    @torch.no_grad()
    def __call__(self, prompt, panel_layout, characters_shot, 
        width=1024, height=1024, batch_size=1,
        num_inference_step=25, seed=42, 
        c_width_given=None, bbox_mode='provided_size_mid_x',
        control_indices=[0], latent_indices=[3],
        crop_face_detect=False, use_face_detect=False, use_rembg=True, 
        cache_results=False, cache_layers=[], 
        use_attention_masking=['no_cross_control_latents', 'mask_control'], 
        debug_name='test1'):
        
        text_kwargs = get_text_preproc(prompt, 
            self.text_encoder1, self.text_encoder2, self.tokenizer1, self.tokenizer2, 
            entity_prompts=[], device=self.device)
        
        control_kwargs, entity_masks, control_nps, debug_mask, print_res = get_all_control_kwargs(
            panel_layout, characters_shot, self.vae, width=width, height=height,
            crop_face_detect=crop_face_detect, use_face_detect=use_face_detect, use_rembg=use_rembg,
            c_width_given=c_width_given, bbox_mode=bbox_mode,
            control_indices=control_indices, latent_indices=latent_indices,
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        total_kwargs = {
            'prompt': prompt, 'sampler': 'unipc', 'width': width, 'height': height, 'frames': 1, 'batch_size': batch_size,
            'num_inference_step': num_inference_step, 'generator': generator, 'device': self.device, 'dtype': self.dtype,
            'cache_results': cache_results, 'cache_layers': cache_layers, 
            'use_attention_masking': use_attention_masking,
            'entity_masks': entity_masks,
        }
        # attn_cache.clear()
        results = sample_hunyuan(transformer=self.model, **total_kwargs, **text_kwargs, **control_kwargs,)

        result_imgs, debug_imgs = [], []
        for i in range(results.shape[0]):
            result_img = Image.fromarray(postproc_imgs(results[i:i+1], self.vae)[0])
            result_imgs.append(result_img)

            meta_str = printable_metadata(total_kwargs, text_kwargs, control_kwargs, self.lora_path, maxlen=80, seed=seed)
            meta_str = meta_str + "\n" + f"batch_num: {i}" + "\n\n" + print_res
            attn_mask = get_pltplot_as_pil(attn_cache['attn_mask'][0], vmin=-9999., vmax=0., cmap=plt.cm.viridis)
            debug_img = return_total_visualization(debug_name, meta_str, np.asarray(result_img), 
                                                    attn_mask, np.asarray(control_nps), np.asarray(debug_mask), 
                                                    np.zeros((height, width, 3), dtype=np.uint8))
            debug_imgs.append(debug_img)
        return result_imgs, debug_imgs, debug_mask