import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from PIL import Image
import torch

def get_text_inds_from_dict(input_str, llama_strtokens):
    start_inds = [i for i, x in llama_strtokens.items() if input_str.startswith(x)]
    end_inds = [i for i, x in llama_strtokens.items() if input_str.endswith(x)]

    if len(start_inds) == 0 or len(end_inds) == 0:
        print("Error")
        return []
    else:
        if len(start_inds) > 1 or len(end_inds) > 1:
            return list(range(start_inds[0], end_inds[-1]+1))
        else:
            return list(range(start_inds[0], end_inds[0]+1))
        
def get_attn_map(attn_cache, attn_inds, block_id=f'transformer_blocks.2', 
                 token_type = 'text',
                 height=960, width=960, token_C=2,
                 embed_size = 729,
                 t_0=0, t_1=25):
    timesteps = sorted(list(attn_cache[list(attn_cache.keys())[0]].keys()), reverse=False)
    token_H, token_W = height // 16, width // 16
    hidden_size = token_H * token_W * token_C
    
    attention_probs = sum(attn_cache[block_id][timesteps[t]] for t in range(t_0, t_1))
    attention_map = attention_probs[:,:,:hidden_size,:]
    attention_map = rearrange(attention_map, 'B A (C H W) D -> B A C H W D', H=token_H, W=token_W)
    attention_map = attention_map[:,:,-1,:,:,:].sum(1).squeeze(1).permute(0,3,1,2) #B, D, H, W
    if token_type == 'text':
        attn_inds = [hidden_size+embed_size+x for x in attn_inds]
    elif token_type == 'image':
        attn_inds = list(range((token_H*token_W*attn_inds[0]), (token_H*token_W*(attn_inds[0]+1))))
    else:
        attn_inds = attn_inds
    print(attention_map.shape)
    attention_data = attention_map[0,attn_inds,:,:].mean(axis=0).to(dtype=torch.float32).cpu().numpy()

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=attention_data.min(), vmax=attention_data.max()) # Normalize data to [0,1] range
    colored_data = cmap(norm(attention_data))[:, :, :3] * 255 # Get RGB values and scale to 0-255
    colored_data_uint8 = colored_data.astype(np.uint8)
    attention_image = Image.fromarray(colored_data_uint8)
    return attention_image

