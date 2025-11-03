import io
import textwrap
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import struct
import cv2
import numpy as np
import pandas as pd

#%%

def convert_to_qwen2vl_format(bbox, h, w):
    x1, y1, x2, y2 = bbox
    x1_new = round(x1 / w * 1000)
    y1_new = round(y1 / h * 1000)
    x2_new = round(x2 / w * 1000)
    y2_new = round(y2 / h * 1000)
    
    x1_new = max(0, min(x1_new, 1000))
    y1_new = max(0, min(y1_new, 1000))
    x2_new = max(0, min(x2_new, 1000))
    y2_new = max(0, min(y2_new, 1000))
    
    return [x1_new, y1_new, x2_new, y2_new]

def convert_from_qwen2vl_format(bbox, w, h):
    x1, y1, x2, y2 = bbox
    x1_new = round(x1 / 1000 * w)
    y1_new = round(y1 / 1000 * h)
    x2_new = round(x2 / 1000 * w)
    y2_new = round(y2 / 1000 * h)

    x1_new = max(0, min(x1_new, w))
    y1_new = max(0, min(y1_new, h))
    x2_new = max(0, min(x2_new, w))
    y2_new = max(0, min(y2_new, h))

    return [x1_new, y1_new, x2_new, y2_new]

def convert_result_to_df_wpose(answer, page_width, page_height):
    pseudo_frame_sub = []
    pseudo_body_sub = []

    for k, v in answer.items():
        bbox = list(map(lambda x: int(x * 1000), v['bbox'])) if v['bbox'][0] < 1 else v['bbox']
        pseudo_frame_sub.append([k] + ['frame'] + bbox)
        for r in v['body']:
            body = list(map(lambda x: int(x * 1000), r[:4])) if r[0] < 1 else r[:4]
            pseudo_body_sub.append([k] + ['body'] + body)

    pseudo_frame_sub = pd.DataFrame(pseudo_frame_sub, columns=['frame_id', 'type', 'rel_xmin', 'rel_ymin', 'rel_xmax', 'rel_ymax'])
    pseudo_frame_sub['order'] = pseudo_frame_sub['frame_id'].apply(lambda x: x.split('-')[1].replace("]", ""))
    pseudo_frame_sub = pseudo_frame_sub.set_index('frame_id')
    pseudo_body_sub = pd.DataFrame(pseudo_body_sub, columns=['frame_id', 'type', 'rel_xmin', 'rel_ymin', 'rel_xmax', 'rel_ymax'])

    for index, row in pseudo_frame_sub.iterrows():
        frame_bbox = [row.rel_xmin, row.rel_ymin, row.rel_xmax, row.rel_ymax]
        frame_bbox = convert_from_qwen2vl_format(frame_bbox, page_width, page_height)
        frame_width, frame_height = frame_bbox[2] - frame_bbox[0], frame_bbox[3] - frame_bbox[1]
        pseudo_frame_sub.loc[index, 'xmin'] = int(frame_bbox[0])
        pseudo_frame_sub.loc[index, 'ymin'] = int(frame_bbox[1])
        pseudo_frame_sub.loc[index, 'xmax'] = int(frame_bbox[2])
        pseudo_frame_sub.loc[index, 'ymax'] = int(frame_bbox[3])

    for index, row in pseudo_body_sub.iterrows():
        parent_row = pseudo_frame_sub.loc[row.frame_id]
        frame_width, frame_height = parent_row.xmax - parent_row.xmin, parent_row.ymax - parent_row.ymin

        body_bbox = [row.rel_xmin, row.rel_ymin, row.rel_xmax, row.rel_ymax]
        body_bbox = convert_from_qwen2vl_format(body_bbox, frame_width, frame_height)
        body_bbox = [
            body_bbox[0]+parent_row.xmin, body_bbox[1]+parent_row.ymin, 
            body_bbox[2]+parent_row.xmin, body_bbox[3]+parent_row.ymin
        ]
        pseudo_body_sub.loc[index, 'xmin'] = int(body_bbox[0])
        pseudo_body_sub.loc[index, 'ymin'] = int(body_bbox[1])
        pseudo_body_sub.loc[index, 'xmax'] = int(body_bbox[2])
        pseudo_body_sub.loc[index, 'ymax'] = int(body_bbox[3])

    page_sub = pd.concat([pseudo_frame_sub, pseudo_body_sub])
    page_sub = page_sub.reset_index().fillna({'order': ''}).astype({'order': 'str'}).rename(columns={'index': 'frame_id2'})
    page_sub.frame_id2 = page_sub.frame_id2.astype(str)
    return page_sub


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    # color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def annot_viz(page_img, page_sub, label_col='', over_img=False):
    color_dict = {"body": "#258039", "face": "#f5be41", "frame": "#31a9b8", "text": "#cf3721"}
    annot_img = np.asarray(page_img).copy() if over_img else np.zeros((page_img.size[1], page_img.size[0], 3), dtype=np.uint8)
    for index, row in page_sub.iterrows():
        bbox = list(map(int, [row.xmin, row.ymin, row.xmax, row.ymax]))
        color = struct.unpack("BBB", bytes.fromhex(color_dict.get(row.type, "#000000")[1:]))
        label = row[label_col] if label_col != '' else ''
        plot_one_box(bbox, annot_img, color=color, label=label)

    return Image.fromarray(annot_img)

def draw_final_img(results_dict, page_w, page_h, key='result_img', panel_width=5):
    final_img = Image.new("RGB", (page_w, page_h), (255,255,255))
    final_draw = ImageDraw.Draw(final_img)
    for panel_id, res_dict in results_dict.items():
        layout = res_dict['panel_layout']
        result_img = res_dict[key]
        x1, y1 = int(layout['bbox'][0]/1000*page_w), int(layout['bbox'][1]/1000*page_h)
        x2, y2 = int(layout['bbox'][2]/1000*page_w), int(layout['bbox'][3]/1000*page_h)
        result_img = result_img.resize((x2-x1, y2-y1))
        final_img.paste(result_img, (x1, y1))
        final_draw.rectangle([x1, y1, x2, y2], outline="black", width=panel_width)
    return final_img

#%%

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

def return_total_visualization(name, meta_str, result_img, attn_mask, control_np, mask_np, gt_np, figsize=(18,10)):
    fig = plt.figure(figsize=figsize)

    ax = plt.subplot2grid((3,4),(0,0),rowspan=2,colspan=2)
    ax.text(0,0.8, meta_str, ha='left', va='top', wrap=True, fontsize=8)
    ax.set_title("Metadata")
    ax.axis('off')

    ax = plt.subplot2grid((3,4),(0,2),rowspan=2,colspan=2)
    ax.imshow(Image.fromarray(result_img))
    ax.set_title("Result Image")
    ax.axis('off')

    ax = plt.subplot2grid((3,4),(2,0))
    ax.imshow(Image.fromarray(mask_np))
    ax.set_title("Target Mask")
    ax.axis('off')

    ax = plt.subplot2grid((3,4),(2,1))
    ax.imshow(Image.fromarray(control_np))
    ax.set_title("Control Image #0")
    ax.axis('off')

    ax = plt.subplot2grid((3,4),(2,2))
    ax.imshow(Image.fromarray(gt_np))
    ax.set_title("GT Image")
    ax.axis('off')

    ax = plt.subplot2grid((3,4),(2,3))
    ax.imshow(attn_mask)
    ax.set_title("Attention Mask")
    ax.axis('off')

    fig.suptitle(f"TestCase: {name}")
    plt.axis("off")
    fig.tight_layout()
    plt.close()

    return fig_to_pil(fig)