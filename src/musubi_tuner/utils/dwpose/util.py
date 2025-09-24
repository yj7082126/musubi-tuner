import math
import numpy as np
import matplotlib
import cv2
import os
from pathlib import Path
from contextlib import suppress
import tempfile
from typing import List, Tuple, Union, Optional
import warnings
from huggingface_hub import constants, hf_hub_download

from .body import BodyResult, Keypoint

eps = 0.01

def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()

UPSCALE_METHODS = ["INTER_NEAREST", "INTER_LINEAR", "INTER_AREA", "INTER_CUBIC", "INTER_LANCZOS4"]
def get_upscale_method(method_str):
    assert method_str in UPSCALE_METHODS, f"Method {method_str} not found in {UPSCALE_METHODS}"
    return getattr(cv2, method_str)

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y
    
def resize_image_with_pad(input_image, resolution, upscale_method = "", skip_hwc3=False, mode='edge'):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    if resolution == 0:
        return img, lambda x: x
    k = float(resolution) / float(min(H_raw, W_raw))
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=get_upscale_method(upscale_method) if k > 1 else cv2.INTER_AREA)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode=mode)

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target, ...])

    return safer_memory(img_padded), remove_pad

def common_input_validate(input_image, output_type, **kwargs):
    if "img" in kwargs:
            warnings.warn("img is deprecated, please use `input_image=...` instead.", DeprecationWarning)
            input_image = kwargs.pop("img")
    
    if "return_pil" in kwargs:
            warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
            output_type = "pil" if kwargs["return_pil"] else "np"
    
    if type(output_type) is bool:
        warnings.warn("Passing `True` or `False` to `output_type` is deprecated and will raise an error in future versions")
        if output_type:
            output_type = "pil"

    if input_image is None:
        raise ValueError("input_image must be defined.")

    if not isinstance(input_image, np.ndarray):
        input_image = np.array(input_image, dtype=np.uint8)
        output_type = output_type or "pil"
    else:
        output_type = output_type or "np"
    
    return (input_image, output_type)

def smart_resize(x, s):
    Ht, Wt = s
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    if Co == 3 or Co == 1:
        k = float(Ht + Wt) / float(Ho + Wo)
        return cv2.resize(x, (int(Wt), int(Ht)), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4)
    else:
        return np.stack([smart_resize(x[:, :, i], s) for i in range(Co)], axis=2)


def smart_resize_k(x, fx, fy):
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    Ht, Wt = Ho * fy, Wo * fx
    if Co == 3 or Co == 1:
        k = float(Ht + Wt) / float(Ho + Wo)
        return cv2.resize(x, (int(Wt), int(Ht)), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4)
    else:
        return np.stack([smart_resize_k(x[:, :, i], fx, fy) for i in range(Co)], axis=2)


def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transfered_model_weights


def is_normalized(keypoints: List[Optional[Keypoint]]) -> bool:
    point_normalized = [
        0 <= abs(k.x) <= 1 and 0 <= abs(k.y) <= 1 
        for k in keypoints 
        if k is not None
    ]
    if not point_normalized:
        return False
    return all(point_normalized)


temp_dir = tempfile.gettempdir()
annotator_ckpts_path = os.path.join(Path(__file__).parents[2], 'ckpts')
USE_SYMLINKS = False
def custom_hf_download(pretrained_model_or_path, filename, cache_dir=temp_dir, ckpts_dir=annotator_ckpts_path, subfolder='', use_symlinks=USE_SYMLINKS, repo_type="model"):

    local_dir = os.path.join(ckpts_dir, pretrained_model_or_path)
    model_path = Path(local_dir).joinpath(*subfolder.split('/'), filename).__str__()

    if len(str(model_path)) >= 255:
        warnings.warn(f"Path {model_path} is too long, \n please change annotator_ckpts_path in config.yaml")

    if not os.path.exists(model_path):
        print(f"Failed to find {model_path}.\n Downloading from huggingface.co")
        print(f"cacher folder is {cache_dir}, you can change it by custom_tmp_path in config.yaml")
        if use_symlinks:
            cache_dir_d = constants.HF_HUB_CACHE    # use huggingface newer env variables `HF_HUB_CACHE`
            if cache_dir_d is None:
                import platform
                if platform.system() == "Windows":
                    cache_dir_d = Path(os.getenv("USERPROFILE")).joinpath(".cache", "huggingface", "hub").__str__()
                else:
                    cache_dir_d = os.path.join(os.getenv("HOME"), ".cache", "huggingface", "hub")
            try:
                # test_link
                Path(cache_dir_d).mkdir(parents=True, exist_ok=True)
                Path(ckpts_dir).mkdir(parents=True, exist_ok=True)
                (Path(cache_dir_d) / f"linktest_{filename}.txt").touch()
                # symlink instead of link avoid `invalid cross-device link` error.
                os.symlink(os.path.join(cache_dir_d, f"linktest_{filename}.txt"), os.path.join(ckpts_dir, f"linktest_{filename}.txt"))
                print("Using symlinks to download models. \n",\
                      "Make sure you have enough space on your cache folder. \n",\
                      "And do not purge the cache folder after downloading.\n",\
                      "Otherwise, you will have to re-download the models every time you run the script.\n",\
                      "You can use USE_SYMLINKS: False in config.yaml to avoid this behavior.")
            except:
                print("Maybe not able to create symlink. Disable using symlinks.")
                use_symlinks = False
                cache_dir_d = Path(cache_dir).joinpath("ckpts", pretrained_model_or_path).__str__()
            finally:    # always remove test link files
                with suppress(FileNotFoundError):
                    os.remove(os.path.join(ckpts_dir, f"linktest_{filename}.txt"))
                    os.remove(os.path.join(cache_dir_d, f"linktest_{filename}.txt"))
        else:
            cache_dir_d = os.path.join(cache_dir, "ckpts", pretrained_model_or_path)

        model_path = hf_hub_download(repo_id=pretrained_model_or_path,
            cache_dir=cache_dir_d,
            local_dir=local_dir,
            subfolder=subfolder,
            filename=filename,
            local_dir_use_symlinks=use_symlinks,
            resume_download=True,
            etag_timeout=100,
            repo_type=repo_type
        )
        if not use_symlinks:
            try:
                import shutil
                shutil.rmtree(os.path.join(cache_dir, "ckpts"))
            except Exception as e :
                print(e)

    print(f"model_path is {model_path}")

    return model_path

def draw_bodypose(canvas: np.ndarray, keypoints: List[Keypoint], xinsr_stick_scaling: bool = False) -> np.ndarray:
    """
    Draw keypoints and limbs representing body pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the body pose.
        keypoints (List[Keypoint]): A list of Keypoint objects representing the body keypoints to be drawn.
        xinsr_stick_scaling (bool): Whether or not scaling stick width for xinsr ControlNet

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn body pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """
    if not is_normalized(keypoints):
        H, W = 1.0, 1.0
    else:
        H, W, _ = canvas.shape

    CH, CW, _ = canvas.shape
    stickwidth = 4

    # Ref: https://huggingface.co/xinsir/controlnet-openpose-sdxl-1.0
    max_side = max(CW, CH)
    if xinsr_stick_scaling:
        stick_scale = 1 if max_side < 500 else min(2 + (max_side // 1000), 7)
    else:
        stick_scale = 1

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], [1, 15], [15, 17], [1, 16], 
        [16, 18],
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None:
            continue

        Y = np.array([keypoint1.x, keypoint2.x]) * float(W)
        X = np.array([keypoint1.y, keypoint2.y]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth*stick_scale), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue

        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)

    return canvas


def draw_handpose(canvas: np.ndarray, keypoints: Union[List[Keypoint], None]) -> np.ndarray:
    """
    Draw keypoints and connections representing hand pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the hand pose.
        keypoints (List[Keypoint]| None): A list of Keypoint objects representing the hand keypoints to be drawn
                                          or None if no keypoints are present.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn hand pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """
    if not keypoints:
        return canvas
    
    if not is_normalized(keypoints):
        H, W = 1.0, 1.0
    else:
        H, W, _ = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for ie, (e1, e2) in enumerate(edges):
        k1 = keypoints[e1]
        k2 = keypoints[e2]
        if k1 is None or k2 is None:
            continue
        
        x1 = int(k1.x * W)
        y1 = int(k1.y * H)
        x2 = int(k2.x * W)
        y2 = int(k2.y * H)
        if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
            cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=2)

    for keypoint in keypoints:
        if keypoint is None:
            continue

        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        if x > eps and y > eps:
            cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(canvas: np.ndarray, keypoints: Union[List[Keypoint], None]) -> np.ndarray:
    """
    Draw keypoints representing face pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the face pose.
        keypoints (List[Keypoint]| None): A list of Keypoint objects representing the face keypoints to be drawn
                                          or None if no keypoints are present.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn face pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """    
    if not keypoints:
        return canvas
    
    if not is_normalized(keypoints):
        H, W = 1.0, 1.0
    else:
        H, W, _ = canvas.shape

    for keypoint in keypoints:
        if keypoint is None:
            continue
        
        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        if x > eps and y > eps:
            cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas


# detect hand according to body pose keypoints
# please refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/hand/handDetector.cpp
def handDetect(body: BodyResult, oriImg) -> List[Tuple[int, int, int, bool]]:
    """
    Detect hands in the input body pose keypoints and calculate the bounding box for each hand.

    Args:
        body (BodyResult): A BodyResult object containing the detected body pose keypoints.
        oriImg (numpy.ndarray): A 3D numpy array representing the original input image.

    Returns:
        List[Tuple[int, int, int, bool]]: A list of tuples, each containing the coordinates (x, y) of the top-left
                                          corner of the bounding box, the width (height) of the bounding box, and
                                          a boolean flag indicating whether the hand is a left hand (True) or a
                                          right hand (False).

    Notes:
        - The width and height of the bounding boxes are equal since the network requires squared input.
        - The minimum bounding box size is 20 pixels.
    """
    ratioWristElbow = 0.33
    detect_result = []
    image_height, image_width = oriImg.shape[0:2]
    
    keypoints = body.keypoints
    # right hand: wrist 4, elbow 3, shoulder 2
    # left hand: wrist 7, elbow 6, shoulder 5
    left_shoulder = keypoints[5]
    left_elbow = keypoints[6]
    left_wrist = keypoints[7]
    right_shoulder = keypoints[2]
    right_elbow = keypoints[3]
    right_wrist = keypoints[4]

    # if any of three not detected
    has_left = all(keypoint is not None for keypoint in (left_shoulder, left_elbow, left_wrist))
    has_right = all(keypoint is not None for keypoint in (right_shoulder, right_elbow, right_wrist))
    if not (has_left or has_right):
        return []
    
    hands = []
    #left hand
    if has_left:
        hands.append([
            left_shoulder.x, left_shoulder.y,
            left_elbow.x, left_elbow.y,
            left_wrist.x, left_wrist.y,
            True
        ])
    # right hand
    if has_right:
        hands.append([
            right_shoulder.x, right_shoulder.y,
            right_elbow.x, right_elbow.y,
            right_wrist.x, right_wrist.y,
            False
        ])

    for x1, y1, x2, y2, x3, y3, is_left in hands:
        # pos_hand = pos_wrist + ratio * (pos_wrist - pos_elbox) = (1 + ratio) * pos_wrist - ratio * pos_elbox
        # handRectangle.x = posePtr[wrist*3] + ratioWristElbow * (posePtr[wrist*3] - posePtr[elbow*3]);
        # handRectangle.y = posePtr[wrist*3+1] + ratioWristElbow * (posePtr[wrist*3+1] - posePtr[elbow*3+1]);
        # const auto distanceWristElbow = getDistance(poseKeypoints, person, wrist, elbow);
        # const auto distanceElbowShoulder = getDistance(poseKeypoints, person, elbow, shoulder);
        # handRectangle.width = 1.5f * fastMax(distanceWristElbow, 0.9f * distanceElbowShoulder);
        x = x3 + ratioWristElbow * (x3 - x2)
        y = y3 + ratioWristElbow * (y3 - y2)
        distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        width = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
        # x-y refers to the center --> offset to topLeft point
        # handRectangle.x -= handRectangle.width / 2.f;
        # handRectangle.y -= handRectangle.height / 2.f;
        x -= width / 2
        y -= width / 2  # width = height
        # overflow the image
        if x < 0: x = 0
        if y < 0: y = 0
        width1 = width
        width2 = width
        if x + width > image_width: width1 = image_width - x
        if y + width > image_height: width2 = image_height - y
        width = min(width1, width2)
        # the max hand box value is 20 pixels
        if width >= 20:
            detect_result.append((int(x), int(y), int(width), is_left))

    '''
    return value: [[x, y, w, True if left hand else False]].
    width=height since the network require squared input.
    x, y is the coordinate of top left 
    '''
    return detect_result


# Written by Lvmin
def faceDetect(body: BodyResult, oriImg) -> Union[Tuple[int, int, int], None]:
    """
    Detect the face in the input body pose keypoints and calculate the bounding box for the face.

    Args:
        body (BodyResult): A BodyResult object containing the detected body pose keypoints.
        oriImg (numpy.ndarray): A 3D numpy array representing the original input image.

    Returns:
        Tuple[int, int, int] | None: A tuple containing the coordinates (x, y) of the top-left corner of the
                                   bounding box and the width (height) of the bounding box, or None if the
                                   face is not detected or the bounding box width is less than 20 pixels.

    Notes:
        - The width and height of the bounding box are equal.
        - The minimum bounding box size is 20 pixels.
    """
    # left right eye ear 14 15 16 17
    image_height, image_width = oriImg.shape[0:2]
    
    keypoints = body.keypoints
    head = keypoints[0]
    left_eye = keypoints[14]
    right_eye = keypoints[15]
    left_ear = keypoints[16]
    right_ear = keypoints[17]
    
    if head is None or all(keypoint is None for keypoint in (left_eye, right_eye, left_ear, right_ear)):
        return None

    width = 0.0
    x0, y0 = head.x, head.y

    if left_eye is not None:
        x1, y1 = left_eye.x, left_eye.y
        d = max(abs(x0 - x1), abs(y0 - y1))
        width = max(width, d * 3.0)

    if right_eye is not None:
        x1, y1 = right_eye.x, right_eye.y
        d = max(abs(x0 - x1), abs(y0 - y1))
        width = max(width, d * 3.0)

    if left_ear is not None:
        x1, y1 = left_ear.x, left_ear.y
        d = max(abs(x0 - x1), abs(y0 - y1))
        width = max(width, d * 1.5)

    if right_ear is not None:
        x1, y1 = right_ear.x, right_ear.y
        d = max(abs(x0 - x1), abs(y0 - y1))
        width = max(width, d * 1.5)

    x, y = x0, y0

    x -= width
    y -= width

    if x < 0:
        x = 0

    if y < 0:
        y = 0

    width1 = width * 2
    width2 = width * 2

    if x + width > image_width:
        width1 = image_width - x

    if y + width > image_height:
        width2 = image_height - y

    width = min(width1, width2)

    if width >= 20:
        return int(x), int(y), int(width)
    else:
        return None


# get max index of 2d array
def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j

def guess_onnx_input_shape_dtype(filename):
    dtype = np.float32
    if "fp16" in filename:
        dtype = np.float16
    elif "int8" in filename:
        dtype = np.uint8
    input_size = (640, 640) if "yolo" in filename else (192, 256)
    if "384" in filename:
        input_size = (288, 384)
    elif "256" in filename:
        input_size = (256, 256)
    return input_size, dtype

if os.getenv('AUX_ORT_PROVIDERS'):
    ONNX_PROVIDERS = os.getenv('AUX_ORT_PROVIDERS').split(',')
else:
    ONNX_PROVIDERS = ["CUDAExecutionProvider", "DirectMLExecutionProvider", "OpenVINOExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"]
def get_ort_providers() -> List[str]:
    providers = []
    try:
        import onnxruntime as ort
        for provider in ONNX_PROVIDERS:
            if provider in ort.get_available_providers():
                providers.append(provider)
        return providers
    except:
        return []

def is_model_torchscript(model) -> bool:
    return bool(type(model).__name__ == "RecursiveScriptModule")

def get_model_type(Nodesname, filename) -> str:
    ort_providers = list(filter(lambda x : x != "CPUExecutionProvider", get_ort_providers()))
    if filename is None:
        return None
    elif ("onnx" in filename) and ort_providers:
        print(f"{Nodesname}: Caching ONNXRuntime session {filename}...")
        return "ort"
    elif ("onnx" in filename):
        print(f"{Nodesname}: Caching OpenCV DNN module {filename} on cv2.DNN...")
        return "cv2"
    else:
        print(f"{Nodesname}: Caching TorchScript module {filename} on ...")
        return  "torchscript"
