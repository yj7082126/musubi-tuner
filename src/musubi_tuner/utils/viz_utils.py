import io
import textwrap
import matplotlib.pyplot as plt
from PIL import Image

def printable_metadata(total_kwargs, text_kwargs, control_kwargs, lora_path, seed=None, maxlen=80):
    keys = [
        'prompt', 'width', 'height', 'frames', 'batch_size', 
        'num_inference_step', 'generator', 
        'use_attention_masking', 'entity_masks', 
        'prompt_embeds',
        'clean_latents', 'latent_indices', 'clean_latent_indices', 'clean_latent_bboxes'
    ]
    meta_content = []
    meta_content.append(f"lora_path: {lora_path}")

    total_kwargs_2 = {**total_kwargs, **text_kwargs, **control_kwargs}.copy()
    for key in keys:
        if key == 'prompt':
            content = total_kwargs_2[key][:(maxlen*4)-10]+"..." if len(total_kwargs_2[key]) > maxlen*4 else total_kwargs_2[key]
        elif key == 'generator':
            content = str(total_kwargs_2[key].seed())
        else:
            content = str(total_kwargs_2.get(key, 'N/A'))
        print_content = '\n'.join(textwrap.wrap(content,maxlen))
        meta_content.append(f"{key}: {print_content}")
    if seed is not None:
        meta_content.append(f"Seed : {seed}")
    return '\n'.join(meta_content)

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return img

def return_total_visualization(name, meta_str, result_img, attn_mask, control_np, mask_np, gt_np):
    fig = plt.figure(figsize=(18,10))

    ax = plt.subplot2grid((3,3),(0,0))
    ax.text(0,0.8, meta_str, ha='left', va='top', wrap=True, fontsize=8)
    ax.set_title("Metadata")
    ax.axis('off')

    ax = plt.subplot2grid((3,3),(1,0))
    ax.imshow(Image.fromarray(mask_np))
    ax.set_title("Target Mask")
    ax.axis('off')

    ax = plt.subplot2grid((3,3),(2,0))
    ax.imshow(Image.fromarray(control_np))
    ax.set_title("Control Image #0")
    ax.axis('off')

    ax = plt.subplot2grid((3,3),(0,1),rowspan=2,colspan=2)
    ax.imshow(Image.fromarray(result_img))
    ax.set_title("Result Image")
    ax.axis('off')

    ax = plt.subplot2grid((3,3),(2,1))
    ax.imshow(Image.fromarray(gt_np))
    ax.set_title("GT Image")
    ax.axis('off')

    ax = plt.subplot2grid((3,3),(2,2))
    ax.imshow(attn_mask)
    ax.set_title("Attention Mask")
    ax.axis('off')

    fig.suptitle(f"TestCase: {name}")
    plt.axis("off")
    fig.tight_layout()
    plt.close()
    
    return fig_to_pil(fig)