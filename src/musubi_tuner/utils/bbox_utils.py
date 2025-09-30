import numpy as np
import copy
from PIL import Image, ImageOps, ImageDraw, PngImagePlugin, JpegImagePlugin
from omegaconf import OmegaConf

def draw_bboxes(img, bboxes, width=2, color=(255, 0, 0)):
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    for bbox in bboxes:
        if bbox is not None:
            x0, y0, x1, y1 = [int(coord * size) for coord, size in zip(bbox, (img_copy.width, img_copy.height, img_copy.width, img_copy.height))]
            draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
    return img_copy

def draw_bboxes_images(img, bboxes, cimg_paths, cimg_sizes=None):
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    for i, (bbox, cimg_path) in enumerate(zip(bboxes, cimg_paths)):
        if bbox is None:
            continue
        x0, y0, x1, y1 = [int(coord * size) for coord, size in zip(bbox, (img_copy.width, img_copy.height, img_copy.width, img_copy.height))]
        if type(cimg_path) not in [Image.Image, np.ndarray, PngImagePlugin.PngImageFile, JpegImagePlugin.JpegImageFile]:
            cimg = Image.open(cimg_path).convert("RGB")
        elif type(cimg_path) == np.ndarray:
            cimg = Image.fromarray(cimg_path).convert("RGB")
        else:
            cimg = cimg_path

        if cimg_sizes is not None:
            cimg_resized = cimg.resize(cimg_sizes[i], Image.LANCZOS)
        else:
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

def scale_layout_data(layout_dict, scale_factor=1.0, direction='top-left'):
    """
    Scale bounding boxes and body keypoints in a specified direction.
    
    Args:
        layout_dict: Dictionary with bbox and body keypoint data
        scale_factor: How much to scale (1.0 = no change, 1.2 = 20% larger, 0.8 = 20% smaller)
        direction: 'top-left', 'top-right', 'bottom-left', 'bottom-right'
    
    Returns:
        Scaled dictionary with same structure
    """
    scaled_dict = copy.deepcopy(layout_dict)
    
    for person_id, person_data in scaled_dict.items():
        bbox = person_data['bbox']  # [x_min, y_min, x_max, y_max]
        body_points = person_data['body']
        
        # Calculate bbox center and dimensions
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Calculate new dimensions
        new_width = width * scale_factor
        new_height = height * scale_factor
        
        # Calculate anchor point based on direction
        if direction == 'top-left':
            anchor_x, anchor_y = x_min, y_min
            new_x_min = anchor_x
            new_y_min = anchor_y
            new_x_max = anchor_x + new_width
            new_y_max = anchor_y + new_height
        elif direction == 'top-right':
            anchor_x, anchor_y = x_max, y_min
            new_x_min = anchor_x - new_width
            new_y_min = anchor_y
            new_x_max = anchor_x
            new_y_max = anchor_y + new_height
        elif direction == 'bottom-left':
            anchor_x, anchor_y = x_min, y_max
            new_x_min = anchor_x
            new_y_min = anchor_y - new_height
            new_x_max = anchor_x + new_width
            new_y_max = anchor_y
        elif direction == 'bottom-right':
            anchor_x, anchor_y = x_max, y_max
            new_x_min = anchor_x - new_width
            new_y_min = anchor_y - new_height
            new_x_max = anchor_x
            new_y_max = anchor_y
        else:
            raise ValueError("Direction must be one of: 'top-left', 'top-right', 'bottom-left', 'bottom-right'")
        
        # Clamp bbox coordinates to valid range [0.0, 1.0]
        new_x_min = max(0.0, min(1.0, new_x_min))
        new_y_min = max(0.0, min(1.0, new_y_min))
        new_x_max = max(0.0, min(1.0, new_x_max))
        new_y_max = max(0.0, min(1.0, new_y_max))
        
        # Update bbox
        scaled_dict[person_id]['bbox'] = [new_x_min, new_y_min, new_x_max, new_y_max]
        
        # Scale body keypoints relative to the same anchor point
        scaled_body_points = []
        for point in body_points:
            point_x, point_y = point
            
            # Calculate offset from anchor point
            offset_x = point_x - anchor_x
            offset_y = point_y - anchor_y
            
            # Scale the offset
            scaled_offset_x = offset_x * scale_factor
            scaled_offset_y = offset_y * scale_factor
            
            # Calculate new point position
            new_point_x = anchor_x + scaled_offset_x
            new_point_y = anchor_y + scaled_offset_y
            
            # Clamp keypoint coordinates to valid range [0.0, 1.0]
            new_point_x = max(0.0, min(1.0, new_point_x))
            new_point_y = max(0.0, min(1.0, new_point_y))
            
            scaled_body_points.append([new_point_x, new_point_y])
        
        scaled_dict[person_id]['body'] = np.array(scaled_body_points)
    
    return scaled_dict

def auto_scale_layout_data(layout_dict, min_area_ratio=0.1, max_area_ratio=0.5, scale_step=0.1):
    """
    Automatically scale bounding boxes and body keypoints to fit naturally within image boundaries.
    
    Args:
        layout_dict: Dictionary with bbox and body keypoint data
        min_area_ratio: Minimum area ratio (0.1 = 10% of image)
        max_area_ratio: Maximum area ratio (0.7 = 70% of image)
        scale_step: Scale factor step size (0.1 = increments of 10%)
    
    Returns:
        Scaled dictionary with same structure, plus metadata about scaling decisions
    """
    import numpy as np
    import copy
    from itertools import product
    
    def bbox_area(bbox):
        """Calculate area of a bbox"""
        x_min, y_min, x_max, y_max = bbox
        return (x_max - x_min) * (y_max - y_min)
    
    def bboxes_overlap(bbox1, bbox2, margin=0.02):
        """Check if two bboxes overlap with a small margin"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Add margin to prevent tight adjacency
        x1_min -= margin; y1_min -= margin; x1_max += margin; y1_max += margin
        
        return not (x1_max <= x2_min or x2_max <= x1_min or y1_max <= y2_min or y2_max <= y1_min)
    
    def is_valid_scaling(layout_dict, person_id, scale_factor, direction):
        """Check if a scaling operation would be valid"""
        person_data = layout_dict[person_id]
        bbox = person_data['bbox']
        
        # Simulate the scaling to check boundaries
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        new_width = width * scale_factor
        new_height = height * scale_factor
        
        # Calculate new bbox based on direction
        if direction == 'top-left':
            anchor_x, anchor_y = x_min, y_min
            new_x_min, new_y_min = anchor_x, anchor_y
            new_x_max, new_y_max = anchor_x + new_width, anchor_y + new_height
        elif direction == 'top-right':
            anchor_x, anchor_y = x_max, y_min
            new_x_min, new_y_min = anchor_x - new_width, anchor_y
            new_x_max, new_y_max = anchor_x, anchor_y + new_height
        elif direction == 'bottom-left':
            anchor_x, anchor_y = x_min, y_max
            new_x_min, new_y_min = anchor_x, anchor_y - new_height
            new_x_max, new_y_max = anchor_x + new_width, anchor_y
        elif direction == 'bottom-right':
            anchor_x, anchor_y = x_max, y_max
            new_x_min, new_y_min = anchor_x - new_width, anchor_y - new_height
            new_x_max, new_y_max = anchor_x, anchor_y
        
        # Check if new bbox stays within boundaries
        if new_x_min < 0.0 or new_y_min < 0.0 or new_x_max > 1.0 or new_y_max > 1.0:
            return False, None
        
        new_bbox = [new_x_min, new_y_min, new_x_max, new_y_max]
        new_area = bbox_area(new_bbox)
        
        # Check area constraints
        if new_area < min_area_ratio or new_area > max_area_ratio:
            return False, None
            
        return True, new_bbox
    
    def find_best_scale_and_direction(layout_dict, person_id, current_scaled_bboxes):
        """Find the best scale factor and direction for a person"""
        current_area = bbox_area(layout_dict[person_id]['bbox'])
        directions = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
        
        # Generate scale factors (0.1 to 3.0 in steps of scale_step)
        max_scale = 3.0
        scale_factors = [round(i * scale_step, 1) for i in range(1, int(max_scale / scale_step) + 1)]
        
        best_options = []
        
        for scale_factor, direction in product(scale_factors, directions):
            # Skip if scale factor is 1.0 (no change)
            if abs(scale_factor - 1.0) < 0.01:
                continue
                
            is_valid, new_bbox = is_valid_scaling(layout_dict, person_id, scale_factor, direction)
            
            if not is_valid:
                continue
            
            # Check for overlaps with already scaled bboxes
            has_overlap = False
            for other_bbox in current_scaled_bboxes:
                if bboxes_overlap(new_bbox, other_bbox):
                    has_overlap = True
                    break
            
            if not has_overlap:
                new_area = bbox_area(new_bbox)
                # Prefer scaling that gets closer to the middle of the allowed area range
                target_area = (min_area_ratio + max_area_ratio) / 2
                area_score = 1.0 - abs(new_area - target_area) / (max_area_ratio - min_area_ratio)
                
                # Prefer moderate scaling over extreme scaling
                scale_score = 1.0 - abs(scale_factor - 1.0) / 2.0
                
                # Combined score
                total_score = area_score * 0.6 + scale_score * 0.4
                
                best_options.append({
                    'scale_factor': scale_factor,
                    'direction': direction,
                    'bbox': new_bbox,
                    'area': new_area,
                    'score': total_score
                })
        
        if not best_options:
            return None
        
        # Sort by score and return the best option
        best_options.sort(key=lambda x: x['score'], reverse=True)
        return best_options[0]
    
    # Start with a copy of the original data
    result_dict = copy.deepcopy(layout_dict)
    scaling_metadata = {}
    scaled_bboxes = []
    
    # Sort people by current area (smallest first, to give them priority for good positions)
    people_by_area = sorted(layout_dict.items(), key=lambda x: bbox_area(x[1]['bbox']))
    
    for person_id, person_data in people_by_area:
        best_option = find_best_scale_and_direction(layout_dict, person_id, scaled_bboxes)
        
        if best_option is None:
            # If no valid scaling found, keep original but record this
            scaling_metadata[person_id] = {
                'scale_factor': 1.0,
                'direction': 'none',
                'status': 'no_valid_scaling_found',
                'original_area': bbox_area(person_data['bbox']),
                'final_area': bbox_area(person_data['bbox'])
            }
            scaled_bboxes.append(person_data['bbox'])
        else:
            # Apply the best scaling found
            scaled_person = scale_layout_data({person_id: person_data}, 
                                            best_option['scale_factor'], 
                                            best_option['direction'])
            
            result_dict[person_id] = scaled_person[person_id]
            scaled_bboxes.append(best_option['bbox'])
            
            scaling_metadata[person_id] = {
                'scale_factor': best_option['scale_factor'],
                'direction': best_option['direction'],
                'status': 'scaled_successfully',
                'original_area': bbox_area(person_data['bbox']),
                'final_area': best_option['area'],
                'score': best_option['score']
            }
    
    return result_dict, scaling_metadata
