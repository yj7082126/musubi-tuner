from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from PIL import Image

from .dwpose import DwposeDetector

dwpose_model = DwposeDetector.from_pretrained(
    '/home/yo564250/workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/hr16/DWPose-TorchScript-BatchSize5',
    '/home/yo564250/workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/hr16/yolox-onnx',
    det_filename='yolox_l.torchscript.pt', 
    pose_filename='dw-ll_ucoco_384_bs5.torchscript.pt',
    torchscript_device='cuda'
)

@dataclass
class Keypoints:
    # Normalized (x,y) in [0,1]
    left_eye: Tuple[float, float]
    right_eye: Tuple[float, float]
    nose: Tuple[float, float]
    neck: Tuple[float, float]

    def to_pixels(self, width: int, height: int) -> Dict[str, np.ndarray]:
        return {
            'left_eye': np.array([self.left_eye[0] * width, self.left_eye[1] * height], dtype=float),
            'right_eye': np.array([self.right_eye[0] * width, self.right_eye[1] * height], dtype=float),
            'nose': np.array([self.nose[0] * width, self.nose[1] * height], dtype=float),
            'neck': np.array([self.neck[0] * width, self.neck[1] * height], dtype=float),
        }

def assemble_points(kp_dict: Dict[str, np.ndarray], use_mid_eye: bool = True) -> np.ndarray:
    pts = [
        kp_dict['left_eye'],
        kp_dict['right_eye'],
        kp_dict['nose'],
        kp_dict['neck'],
    ]
    if use_mid_eye:
        mid_eye = 0.5 * (kp_dict['left_eye'] + kp_dict['right_eye'])
        pts.append(mid_eye)
    return np.stack(pts, axis=0)  # shape (N,2)

def solve_scale_translation(src_pts: np.ndarray, tgt_pts: np.ndarray):
    """
    Least squares solution for s, t (no rotation).
    src_pts, tgt_pts: (N,2)
    Returns scale (float), translation (2,)
    """
    assert src_pts.shape == tgt_pts.shape and src_pts.ndim == 2 and src_pts.shape[1] == 2
    src_mean = src_pts.mean(axis=0)
    tgt_mean = tgt_pts.mean(axis=0)
    src_c = src_pts - src_mean
    tgt_c = tgt_pts - tgt_mean

    denom = np.sum(np.sum(src_c * src_c, axis=1))  # Σ ||X'_i||²
    if denom < 1e-8:
        scale = 1.0
    else:
        # Σ (X'_i · Y'_i)
        numer = np.sum(np.sum(src_c * tgt_c, axis=1))
        scale = numer / denom if denom > 0 else 1.0

    translation = tgt_mean - scale * src_mean
    return scale, translation  # translation is (dx, dy)

def rescale_trans(scale_2, bbox, base_scale, base_translation, src_pts, sw, sh, tw, th):
    assert scale_2 is not None or bbox is not None, "Either scale_2 or bbox must be provided"
    # Handle scale_2 parameter
    if scale_2 is None and bbox is not None:
        # Auto-infer scale_2 to fit within bbox
        bbox_w = (bbox[2] - bbox[0]) * tw
        bbox_h = (bbox[3] - bbox[1]) * th
        
        # Calculate what the source image size would be with base_scale
        scaled_sw = sw * base_scale
        scaled_sh = sh * base_scale
        
        # Find the scale_2 that fits within bbox (multiples of 0.5)
        max_scale_w = bbox_w / scaled_sw if scaled_sw > 0 else 1.0
        max_scale_h = bbox_h / scaled_sh if scaled_sh > 0 else 1.0
        max_scale = min(max_scale_w, max_scale_h)
        
        # Find largest multiple of 0.5 that doesn't exceed max_scale
        scale_2 = 0.5
        while scale_2 * 2 <= max_scale:
            scale_2 *= 2
        if scale_2 > max_scale:
            scale_2 = 0.5
    elif scale_2 is None:
        scale_2 = 1.0
    
    final_scale = base_scale * scale_2
    
    # Correct translation to maintain center alignment
    # When we multiply scale by scale_2, the image expands around origin (0,0)
    # We need to adjust translation to keep the keypoints centered
    scale_correction = scale_2 - 1.0
    
    # Calculate the center of the transformed keypoints with base scale
    transformed_center = base_translation + base_scale * src_pts.mean(axis=0)
    
    # Adjust translation to compensate for the additional scaling
    correction = -scale_correction * base_scale * src_pts.mean(axis=0)
    final_translation = base_translation + correction
    
    # If bbox is provided, adjust translation to center within bbox
    if bbox is not None:
        bbox_center_x = (bbox[0] + bbox[2]) * 0.5 * tw
        bbox_center_y = (bbox[1] + bbox[3]) * 0.5 * th
        
        # Calculate where the source center would be with current transformation
        source_center = final_translation + final_scale * np.array([sw/2, sh/2])
        
        # Adjust translation to align source center with bbox center
        center_offset = np.array([bbox_center_x, bbox_center_y]) - source_center
        final_translation = final_translation + center_offset
    return final_scale, final_translation

def transform_and_paste(
    source_img: Image.Image,
    scale: float,
    translation: np.ndarray,
    target_size: Tuple[int, int],
    fill_color=(0,0,0)
) -> Image.Image:
    """
    Uniformly scale source image around its origin (top-left), then translate,
    and composite onto a blank target canvas of size target_size (w,h).
    """
    tw, th = target_size
    canvas = Image.new(source_img.mode, (tw, th), fill_color)

    # Scale
    sw, sh = source_img.size
    new_w, new_h = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
    scaled = source_img.resize((new_w, new_h), Image.BICUBIC)

    x0, y0 = int(round(translation[0])), int(round(translation[1]))
    # Clip destination
    dx0, dy0 = max(x0, 0), max(y0, 0)
    dx1, dy1 = min(x0 + new_w, tw),  min(y0 + new_h, th)
    if dx1 <= dx0 or dy1 <= dy0:
        return canvas  # completely out of frame

    # Corresponding source crop (if part goes outside)
    crop_left = (dx0 - x0)
    crop_top  = (dy0 - y0)
    crop_right = crop_left + (dx1 - dx0)
    crop_bottom = crop_top + (dy1 - dy0)

    cropped = scaled.crop((crop_left, crop_top, crop_right, crop_bottom))
    bbox = (dx0/tw, dy0/th, dx1/tw, dy1/th)
    canvas.paste(cropped, (dx0, dy0))
    return canvas, bbox

def align_source_to_target(
    source_img: Image.Image,
    source_kps: Keypoints,
    source_size: Tuple[int,int],   # (w,h) of source image (redundant but explicit)
    target_kps: Keypoints,
    target_size: Tuple[int,int],   # (w,h) of target frame
    use_mid_eye: bool = True,
    scale_2: float = 1.0,
    bbox: Tuple[float, float, float, float] = None  # (x0, y0, x1, y1) in [0,1]
):
    """
    Returns:
        aligned_image (PIL.Image)
        bbox_result (tuple): actual bounding box of the placed image
        scale (float): final scale applied
        translation (np.ndarray of shape (2,)): final translation applied
    """
    sw, sh = source_size
    tw, th = target_size

    src_px = source_kps.to_pixels(sw, sh)
    tgt_px = target_kps.to_pixels(tw, th)

    src_pts = assemble_points(src_px, use_mid_eye=use_mid_eye)
    tgt_pts = assemble_points(tgt_px, use_mid_eye=use_mid_eye)

    base_scale, base_translation = solve_scale_translation(src_pts, tgt_pts)
    final_scale, final_translation = rescale_trans(scale_2, bbox, base_scale, base_translation, src_pts, sw, sh, tw, th)
    aligned, bbox_result = transform_and_paste(source_img, final_scale, final_translation, target_size)
    return aligned, bbox_result, final_scale, final_translation

def detect_and_align_source_to_target(control_image_path, target_kps, target_size, scale_2=1.0, bbox=None):
    source_img = Image.open(control_image_path).convert("RGB")
    source_size = np.array(source_img.size)
    out_img, out_json = dwpose_model(source_img, image_and_json=True)
    if len(out_json['people']) > 0:
        src_kps = out_json['people'][0]['pose_keypoints_2d']
        src_kps = np.array(src_kps).reshape(-1,3)[[0,1,14,15],:2]
        src_kps = src_kps / source_size[None,:]
        src_kps = Keypoints(left_eye=src_kps[2], right_eye=src_kps[3], nose=src_kps[0], neck=src_kps[1])
        aligned_img, face_bbox, scale, translation = align_source_to_target(
            source_img, src_kps, source_size,
            target_kps, target_size,
            use_mid_eye=True, scale_2=scale_2,
            bbox=None
        )
        return face_bbox
    else:
        return bbox
    
def search_facebbox_for_layout(panel_layout, characters_shot, target_size, scale=1.0):
    debug_dict = {}
    for k, v in panel_layout.items():
        debug_els = {}
        tgt_kps = Keypoints(
            left_eye=v['body'][2], right_eye=v['body'][3], 
            nose=v['body'][0], neck=v['body'][1]
        )
        debug_els['entity_bbox'] = v['bbox']

        control_image_path = characters_shot[list(characters_shot.keys())[k]]['images'][0]
        debug_els['control_image_path'] = control_image_path
        face_bbox = detect_and_align_source_to_target(
            control_image_path, tgt_kps, target_size, scale_2=scale, bbox=v['bbox'])

        debug_els['face_bbox'] = face_bbox
        debug_dict[k] = debug_els
    return debug_dict