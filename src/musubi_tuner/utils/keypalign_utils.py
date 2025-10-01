from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from PIL import Image
from torch import device

from .dwpose import DwposeDetector
from .bbox_utils import get_facebbox_from_bbox

dwpose_model = DwposeDetector.from_pretrained(
    'hr16/DWPose-TorchScript-BatchSize5',
    'hr16/yolox-onnx',
    det_filename='yolox_l.torchscript.pt', 
    pose_filename='dw-ll_ucoco_384_bs5.torchscript.pt',
    torchscript_device='cuda'
)

from facexlib.utils.face_restoration_helper import FaceRestoreHelper

face_helper = FaceRestoreHelper(
    upscale_factor=1,
    face_size=512,
    crop_ratio=(1, 1),
    det_model='retinaface_resnet50',
    save_ext='png',
    device='cuda',
)
face_helper.face_det.to('cuda')
face_helper.face_parse.to('cuda')

def is_valid_keypoint(keypoints):
    """Return valid keypoint mask (N,) for keypoint array (N,2) based on mean+std threshold"""
    mean, std = keypoints.mean(0), keypoints.std(0)
    return ((keypoints >= mean - std) & (keypoints <= mean + std)).all(1)

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

def safety_scale(scale, sw, sh, tw, th, min_boundary=0.02, max_boundary=0.2):
    scaled_sw = sw * scale
    scaled_sh = sh * scale
    s_ratio = scaled_sw * scaled_sh / (tw * th)
    if s_ratio < min_boundary:
        new_scale = 1.0
        i = 0
        while s_ratio < min_boundary and i < 10:
            new_scale += 0.5
            scaled_sw = sw * (scale * new_scale)
            scaled_sh = sh * (scale * new_scale)
            s_ratio = scaled_sw * scaled_sh / (tw * th)
            i += 1
        return new_scale
    elif s_ratio > max_boundary:
        new_scale = 1.0
        while s_ratio > max_boundary and new_scale > 0:
            new_scale -= 0.1
            scaled_sw = sw * (scale * new_scale)
            scaled_sh = sh * (scale * new_scale)
            s_ratio = scaled_sw * scaled_sh / (tw * th)
        return new_scale if new_scale > 0 else 0.5
    else:
        return 1.0

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
    source_kps: np.ndarray,
    source_size: Tuple[int,int],   # (w,h) of source image (redundant but explicit)
    target_kps: np.ndarray,
    target_size: Tuple[int,int],   # (w,h) of target frame
    min_boundary: float = 0.02,
    max_boundary: float = 0.2,
    bbox: Tuple[float, float, float, float] = None  # (x0, y0, x1, y1) in [0,1]
):
    """
    Returns:
        aligned_image (PIL.Image)
        bbox_result (tuple): actual bounding box of the placed image
        scale (float): final scale applied
        translation (np.ndarray of shape (2,)): final translation applied
    """
    base_scale, base_translation = solve_scale_translation(source_kps, target_kps)
    scale_2 = safety_scale(base_scale, source_size[0], source_size[1], target_size[0], target_size[1], min_boundary=min_boundary, max_boundary=max_boundary)
    final_scale, final_translation = rescale_trans(scale_2, bbox, base_scale, base_translation, source_kps, source_size[0], source_size[1], target_size[0], target_size[1])
    aligned, bbox_result = transform_and_paste(source_img, final_scale, final_translation, target_size)
    return aligned, bbox_result, final_scale, final_translation

def detect_and_align_source_to_target(source_img, target_kps, target_size, 
                                    min_bbox_size=0.02, max_bbox_size=0.2, bbox=None):
    source_size = np.array(source_img.size)
    out_img, out_json = dwpose_model(source_img, image_and_json=True)
    if len(out_json['people']) > 0:
        src_kps = out_json['people'][0]['pose_keypoints_2d']
        src_kps = np.array(src_kps).reshape(-1,3)[[0,1,14,15]]
        valid_src_keyps = src_kps[:,-1].astype(bool) & is_valid_keypoint(src_kps[:,:2])
        valid_tgt_keyps = is_valid_keypoint(target_kps)
        valid_keyps = valid_src_keyps & valid_tgt_keyps
        src_kps = src_kps[valid_keyps,:2] 
        tgt_kps = np.array(target_kps)[valid_keyps,:] * np.array(target_size)  # only use the valid keypoints in source

        if (len(src_kps) == 0) or (len(tgt_kps) == 0):
            return None
        else:
            aligned_img, face_bbox, scale, translation = align_source_to_target(
                source_img, src_kps, source_size,
                tgt_kps, target_size,
                min_boundary=min_bbox_size,
                max_boundary=max_bbox_size,
                bbox=None
            )
            return face_bbox
    else:
        return None
    
def align_face(control_image):
    face_helper.clean_all()
    face_helper.read_image(np.array(control_image)[...,::-1])
    face_helper.get_face_landmarks_5(only_center_face=True)
    face_helper.align_warp_face()
    if len(face_helper.cropped_faces) == 0:
        return control_image
    align_face = face_helper.cropped_faces[0]
    return Image.fromarray(align_face[...,::-1])

def search_facebbox_for_layout(
        panel_layout, characters_shot, target_size, 
        crop_face_detect=True, use_face_detect=True,
        c_width_given=None, bbox_mode="full_width_relative_height", 
        min_bbox_size=0.04, max_bbox_size=0.2, 
        scale_c=1.2, use_safety=True, min_safety_ratio=0.1, max_safety_ratio=0.4
        ):
    """
    Used to calculate the respective face bboxes (for RoPE) and the control image sizes, 
    given the information in panel_layout and characters_shot.


    Args:
        panel_layout (dict): layout data for each character, including 'bbox' and optionally 'body' keypoints
        characters_shot (dict): character data including image paths
        target_size (tuple): (width, height) of the target canvas
        use_face_detect (bool): whether to use face detection to refine control image size
        c_width_given (int or None): if provided, use this width for control image size

    Procedure:
    0. If crop_face_detect is True, preprocess the control image to align face first.
    * This should help for cases where the character image is a full-body, impairing plausible content generation. (E.g ViStory-5)
    * If detect / align fails, fall back to using the original image.

    1. If use_face_detect is True and body keypoints are provided, use them to detect face bbox.
        * Bbox_scale is used to scale the detected face bbox, fixing cases where the bbox is too small.
        * If there is a face bbox detected, use the detected face bbox to adjust control image size,
          so that the control image is approximately X% of the face bbox size. 
          (Control by ratio_1)
        * If no face bbox is detected, go to step 2
        * If there exists c_width_given, override and use it to determine the control image size
    2. If use_face_detect is False or no body keypoints are provided, calculate the face bbox and control image size differently
        * If c_width_given is provided, use it to determine the control image size

    """
    debug_dict = {}
    if len(list(characters_shot.keys())) > 0:
        for k, v in panel_layout.items():
            debug_els = {}
            debug_els['entity_bbox'] = v['bbox']

            character_name = list(characters_shot.keys())[k] if len(list(characters_shot.keys())) > k else list(characters_shot.keys())[0]
            control_image_path = characters_shot[character_name]['images'][0]        
            control_image = Image.open(control_image_path)
            if crop_face_detect:
                control_image = align_face(control_image)
            if c_width_given is not None:
                control_ratio = control_image.size[1] / control_image.size[0]
                c_width = int((c_width_given) // 16 * 16)
                c_height = int(round(c_width * control_ratio) // 2 * 2)
                control_image = control_image.resize((c_width, c_height), Image.LANCZOS)

            control_ratio = control_image.size[1] / control_image.size[0]
            debug_els['control_image_path'] = control_image_path
            debug_els['control_image'] = control_image

            # if scale_c_by_bbox:
            #     c_width = int(target_size[0] * (v['bbox'][2]-v['bbox'][0]) // 16 * 16)
            # else:
            #     c_width = int(np.sqrt(target_size[0] * target_size[1] * scale_c) // 16 * 16)
            # c_width, c_height = int(c_width * scale), int(c_height * scale) 

            face_bbox = None
            if 'body' in v and len(v['body']) == 4 and use_face_detect:
                tgt_kps = v['body']
                face_bbox = detect_and_align_source_to_target(
                    control_image, tgt_kps, target_size, 
                    min_bbox_size=min_bbox_size, max_bbox_size=max_bbox_size,
                    bbox=v['bbox'])
            if face_bbox is None:
                face_bbox = v['body'][:4] if len(v['body']) >= 4 else None
                face_bbox = get_facebbox_from_bbox(
                    v['bbox'], control_image.size[0], control_image.size[1], 
                    target_size[0], target_size[1],
                    face_bbox=face_bbox, mode=bbox_mode)
            
            if c_width_given is None:
                c_width = int((target_size[0] * (face_bbox[2]-face_bbox[0])) * scale_c // 16 * 16)
                c_height = int(round(c_width * control_ratio) // 2 * 2)  
            if use_safety:
                if (c_width * c_height) < (target_size[0] * target_size[1] * min_safety_ratio):
                    c_width = (target_size[0] * target_size[1] * min_safety_ratio / control_ratio) ** 0.5
                    c_width = int(c_width // 16 * 16)
                    c_height = int(round(c_width * control_ratio) // 2 * 2)
                elif (c_width * c_height) > (target_size[0] * target_size[1] * max_safety_ratio):
                    c_width = (target_size[0] * target_size[1] * max_safety_ratio / control_ratio) ** 0.5
                    c_width = int(c_width // 16 * 16)
                    c_height = int(round(c_width * control_ratio) // 2 * 2)

            debug_els['control_image_size'] = (c_width, c_height)
            debug_els['face_bbox'] = face_bbox

            debug_dict[k] = debug_els
    return debug_dict

    # if len(debug_dict) == 0:
    #     control_image_paths = []
    #     control_image_sizes = []
    #     entity_bboxes = []
    #     face_bboxes = []
    # else:
    #     control_image_paths = [debug_dict[i]['control_image_path'] for i in range(max_characters)]
    #     control_image_sizes = [debug_dict[i]['control_image_size'] for i in range(max_characters)]
    #     entity_bboxes = [debug_dict[i]['entity_bbox'] for i in range(max_characters)]
    #     face_bboxes = [debug_dict[i]['face_bbox'] for i in range(max_characters)]

    # return {
    #     'debug_dict': debug_dict,
    #     'control_image_paths': control_image_paths,
    #     'control_image_sizes': control_image_sizes,
    #     'entity_bboxes': entity_bboxes,
    #     'face_bboxes': face_bboxes
    # }