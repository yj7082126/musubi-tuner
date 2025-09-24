import numpy as np
from PIL import Image, ImageOps, ImageDraw
from omegaconf import OmegaConf

def draw_bboxes(img, bboxes, width=2, color=(255, 0, 0)):
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(coord * size) for coord, size in zip(bbox, (img_copy.width, img_copy.height, img_copy.width, img_copy.height))]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
    return img_copy

def draw_bboxes_images(img, bboxes, cimgs):
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    for bbox, cimg in zip(bboxes, cimgs):
        x0, y0, x1, y1 = [int(coord * size) for coord, size in zip(bbox, (img_copy.width, img_copy.height, img_copy.width, img_copy.height))]
        cimg_resized = cimg.resize((x1 - x0, y1 - y0), Image.LANCZOS)
        img_copy.paste(cimg_resized, (x0, y0))
    return img_copy

def get_mask_from_bboxes(bboxes, width, height):
    newimg = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(newimg)
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(coord * size) for coord, size in zip(bbox, (width, height, width, height))]
        draw.rectangle([x0, y0, x1, y1], outline=255, fill=255)
    return np.array(newimg)

def get_bbox_from_mask(mask):
    h, w = mask.shape[:2]
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = xs.min().item(), xs.max().item()
    y0, y1 = ys.min().item(), ys.max().item()
    return [x0 / w, y0 / h, x1 / w, y1 / h]

def get_bbox_from_str(bbox_str, width=512, height=512):
    bbox_vals = [v.strip() for v in bbox_str.replace(';', ',').split(',') if v.strip()]
    floats = list(map(float, bbox_vals))
    # If any value >1 treat as pixel coordinates and normalize
    if any(v > 1 for v in floats):
        x1, y1, x2, y2 = floats
        floats = [x1/width, y1/height, x2/width, y2/height]
    return floats


# def calc_latent_bbox(bbox, c_img_tensor, width, height):
#     c_H, c_W = c_img_tensor.shape[-2:]
#     entity_h, entity_w = c_H / height, c_W / width
#     latent_bbox = [bbox[0], bbox[1], bbox[0]+entity_w, bbox[1]+entity_h]
#     return latent_bbox


def get_facebbox_from_bbox(bbox, c_W, c_H, w, h, face_bbox=None, mode="full_width_relative_height"):
    assert mode in [
        "provided_face_bbox",
        "full_width_full_height", 
        "full_width_relative_height",
        "relative_width_full_height",
        "provided_size_mid_x"
    ]
    if mode == "provided_face_bbox":
        assert face_bbox is not None, "face_bbox must be provided in 'provided_face_bbox' mode"
        return face_bbox

    p_bbox = [bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h]
    
    if mode == "full_width_full_height":
        face_bbox = [p_bbox[0], p_bbox[1], p_bbox[2], p_bbox[3]]
    elif mode == "full_width_relative_height":
        face_bbox = [
            p_bbox[0], p_bbox[1], p_bbox[2],
            min(p_bbox[1]+(p_bbox[2]-p_bbox[0])*(c_H/c_W), p_bbox[3])
        ]  
    elif mode == "relative_width_full_height":
        face_bbox = [
            p_bbox[0], p_bbox[1], 
            min(p_bbox[0]+(p_bbox[3]-p_bbox[1])*(c_W/c_H), p_bbox[2]),
            p_bbox[3]
        ]
    elif mode == "provided_size_mid_x":
        # face_bbox = [
        #     p_bbox[0], p_bbox[1], 
        #     min(p_bbox[0]+c_W, p_bbox[2]),
        #     min(p_bbox[1]+c_H, p_bbox[3])
        # ]
        p_bbox_w_mid = (p_bbox[0] + p_bbox[2]) / 2
        face_bbox = [
            max(p_bbox_w_mid-(c_W//2), p_bbox[0]), 
            p_bbox[1], 
            min(p_bbox_w_mid+(c_W//2), p_bbox[2]),
            min(p_bbox[1]+c_H, p_bbox[3])
        ]
    else:
        raise NotImplementedError("mode not implemented")

    face_bbox = [face_bbox[0]/w, face_bbox[1]/h, face_bbox[2]/w, face_bbox[3]/h]
    return face_bbox

def get_bbox_from_meta(meta_path, len_control):
    meta = OmegaConf.load(meta_path)
    bboxes = []
    for key, int_bbox in meta['target_body'].items():
        rel_bbox = [int_bbox[0]/meta['width'], int_bbox[1]/meta['height'], int_bbox[2]/meta['width'], int_bbox[3]/meta['height']]
        bboxes.append(rel_bbox)

    if len(bboxes) < len_control:
        for _ in range(len_control - len(bboxes)):
            bboxes.append([-1.0, -1.0, -1.0, -1.0]) # pad with invalid bbox
    elif len(bboxes) > len_control:
        bboxes = bboxes[:len_control] # truncate if more than control count
    return bboxes
