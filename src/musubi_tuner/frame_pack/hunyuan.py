# original code: https://github.com/lllyasviel/FramePack
# original license: Apache-2.0

import torch

# from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE
# from diffusers_helper.utils import crop_or_pad_yield_mask
from musubi_tuner.hunyuan_model.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from musubi_tuner.hunyuan_model.text_encoder import PROMPT_TEMPLATE


@torch.no_grad()
def encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2, max_length=256, custom_system_prompt=None, return_tokendict=False):
    assert isinstance(prompt, str)

    prompt = [prompt]

    # LLAMA

    # We can verify crop_start by checking the token count of the prompt:
    # custom_system_prompt = (
    #     "Describe the video by detailing the following aspects: "
    #     "1. The main content and theme of the video."
    #     "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    #     "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    #     "4. background environment, light, style and atmosphere."
    #     "5. camera angles, movements, and transitions used in the video:"
    # )
    if custom_system_prompt is None:
        prompt_llama = [PROMPT_TEMPLATE["dit-llm-encode-video"]["template"].format(p) for p in prompt]
        crop_start = PROMPT_TEMPLATE["dit-llm-encode-video"]["crop_start"]
    else:
        # count tokens for custom_system_prompt
        full_prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{custom_system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        print(f"Custom system prompt: {full_prompt}")
        system_prompt_tokens = tokenizer(full_prompt, return_tensors="pt", truncation=True).input_ids[0].shape[0]
        print(f"Custom system prompt token count: {system_prompt_tokens}")
        prompt_llama = [full_prompt + p + "<|eot_id|>" for p in prompt]
        crop_start = system_prompt_tokens

    llama_inputs = tokenizer(
        prompt_llama,
        padding="max_length",
        max_length=max_length + crop_start,
        truncation=True,
        return_tensors="pt",
        return_length=False,
        return_overflowing_tokens=False,
        return_attention_mask=True,
    )

    llama_input_ids = llama_inputs.input_ids.to(text_encoder.device)
    llama_attention_mask = llama_inputs.attention_mask.to(text_encoder.device)
    llama_attention_length = int(llama_attention_mask.sum())

    llama_outputs = text_encoder(
        input_ids=llama_input_ids,
        attention_mask=llama_attention_mask,
        output_hidden_states=True,
    )

    llama_vec = llama_outputs.hidden_states[-3][:, crop_start:llama_attention_length]
    # llama_vec_remaining = llama_outputs.hidden_states[-3][:, llama_attention_length:]
    llama_attention_mask = llama_attention_mask[:, crop_start:llama_attention_length]

    assert torch.all(llama_attention_mask.bool())

    llama_tokens = llama_inputs.input_ids[0,crop_start:llama_attention_length].cpu().numpy()
    llama_strtokens = [x.replace('Ġ', '') for x in tokenizer.convert_ids_to_tokens(llama_tokens)]
    llama_strtokens = {i: x for i,x in enumerate(llama_strtokens)}

    # CLIP

    clip_l_input_ids = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    ).input_ids
    clip_l_pooler = text_encoder_2(clip_l_input_ids.to(text_encoder_2.device), output_hidden_states=False).pooler_output

    if return_tokendict:
        return llama_vec, clip_l_pooler, llama_strtokens
    else:
        return llama_vec, clip_l_pooler


@torch.no_grad()
def vae_decode_fake(latents):
    latent_rgb_factors = [
        [-0.0395, -0.0331, 0.0445],
        [0.0696, 0.0795, 0.0518],
        [0.0135, -0.0945, -0.0282],
        [0.0108, -0.0250, -0.0765],
        [-0.0209, 0.0032, 0.0224],
        [-0.0804, -0.0254, -0.0639],
        [-0.0991, 0.0271, -0.0669],
        [-0.0646, -0.0422, -0.0400],
        [-0.0696, -0.0595, -0.0894],
        [-0.0799, -0.0208, -0.0375],
        [0.1166, 0.1627, 0.0962],
        [0.1165, 0.0432, 0.0407],
        [-0.2315, -0.1920, -0.1355],
        [-0.0270, 0.0401, -0.0821],
        [-0.0616, -0.0997, -0.0727],
        [0.0249, -0.0469, -0.1703],
    ]  # From comfyui

    latent_rgb_factors_bias = [0.0259, -0.0192, -0.0761]

    weight = torch.tensor(latent_rgb_factors, device=latents.device, dtype=latents.dtype).transpose(0, 1)[:, :, None, None, None]
    bias = torch.tensor(latent_rgb_factors_bias, device=latents.device, dtype=latents.dtype)

    images = torch.nn.functional.conv3d(latents, weight, bias=bias, stride=1, padding=0, dilation=1, groups=1)
    images = images.clamp(0.0, 1.0)

    return images


@torch.no_grad()
def vae_decode(latents, vae, image_mode=False) -> torch.Tensor:
    latents = latents / vae.config.scaling_factor

    if not image_mode:
        image = vae.decode(latents.to(device=vae.device, dtype=vae.dtype)).sample
    else:
        latents = latents.to(device=vae.device, dtype=vae.dtype).unbind(2)
        image = [vae.decode(l.unsqueeze(2)).sample for l in latents]
        image = torch.cat(image, dim=2)

    return image


@torch.no_grad()
def vae_encode(image, vae: AutoencoderKLCausal3D) -> torch.Tensor:
    latents = vae.encode(image.to(device=vae.device, dtype=vae.dtype)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    return latents
