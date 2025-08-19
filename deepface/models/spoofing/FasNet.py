# built-in dependencies
from typing import Union

# 3rd party dependencies
import cv2
import numpy as np

# project dependencies
from deepface.commons import weight_utils
from deepface.commons.logger import Logger

logger = Logger()

# pylint: disable=line-too-long, too-few-public-methods, nested-min-max
FIRST_WEIGHTS_URL="https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
# SECOND_WEIGHTS_URL="https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth"
SECOND_WEIGHTS_URL="https://github.com/hairymax/Face-AntiSpoofing/raw/refs/heads/main/saved_models/AntiSpoofing_bin_1.5_128.pth"

class Fasnet:
    """
    Mini Face Anti Spoofing Net Library from repo: github.com/minivision-ai/Silent-Face-Anti-Spoofing

    Minivision's Silent-Face-Anti-Spoofing Repo licensed under Apache License 2.0
    Ref: github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/src/model_lib/MiniFASNet.py
    """

    def __init__(self):
        # pytorch is an opitonal dependency, enforce it to be installed if class imported
        try:
            import torch
        except Exception as err:
            raise ValueError(
                "You must install torch with `pip install torch` command to use face anti spoofing module"
            ) from err

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        # download pre-trained models if not installed yet
        first_model_weight_file = weight_utils.download_weights_if_necessary(
            file_name="2.7_80x80_MiniFASNetV2.pth",
            source_url=FIRST_WEIGHTS_URL,
        )

        second_model_weight_file = weight_utils.download_weights_if_necessary(
            file_name="AntiSpoofing_bin_1.5_128.pth",
            source_url=SECOND_WEIGHTS_URL,
        )

        # guarantees Fasnet imported and torch installed
        from deepface.models.spoofing import FasNetBackbone

        # Create custom configuration for the AntiSpoofing_bin_1.5_128.pth weights
        # This configuration matches the channel dimensions found in the checkpoint
        def create_custom_minifasnet_se():
            """Create a MiniFASNetSE with configuration matching AntiSpoofing_bin_1.5_128.pth"""
            # Custom keep_dict extracted directly from checkpoint weights
            # Format: [conv_23_exp, conv_23_exp, conv_3.0_exp, conv_3.0_exp, conv_3.1_exp, conv_3.1_exp, ...]
            custom_keep_dict = [
                32, 32,       # Initial conv layers (conv1: 32, conv2_dw: 32)
                103, 103,     # Conv_23 expansion: 103 channels
                64,           # Conv_3 input channels
                13, 13,       # Conv_3.0 expansion: 13 channels
                64,           # Conv_3.1 input channels
                13, 13,       # Conv_3.1 expansion: 13 channels  
                64,           # Conv_3.2 input channels
                13, 13,       # Conv_3.2 expansion: 13 channels
                64,           # Conv_3.3 input channels
                13, 13,       # Conv_3.3 expansion: 13 channels
                64,           # Conv_34 input channels
                231, 231,     # Conv_34 expansion: 231 channels
                128,          # Conv_4 input channels
                231, 231,     # Conv_4.0 expansion: 231 channels
                128,          # Conv_4.1 input channels
                52, 52,       # Conv_4.1 expansion: 52 channels
                128,          # Conv_4.2 input channels
                26, 26,       # Conv_4.2 expansion: 26 channels
                128,          # Conv_4.3 input channels
                77, 77,       # Conv_4.3 expansion: 77 channels
                128,          # Conv_4.4 input channels
                26, 26,       # Conv_4.4 expansion: 26 channels
                128,          # Conv_4.5 input channels
                26, 26,       # Conv_4.5 expansion: 26 channels
                128,          # Conv_45 input channels
                308, 308,     # Conv_45 expansion: 308 channels
                128,          # Conv_5 input channels
                26, 26,       # Conv_5.0 expansion: 26 channels
                128,          # Conv_5.1 input channels
                26, 26,       # Conv_5.1 expansion: 26 channels
                128,          # Conv_6_sep input channels
                512, 512,     # Conv_6_sep and Conv_6_dw: 512 channels
            ]
            
            return FasNetBackbone.MiniFASNetSE(
                custom_keep_dict, 
                embedding_size=128, 
                conv6_kernel=(8, 8),  # Matches checkpoint kernel size
                drop_p=0.75, 
                num_classes=2,        # Matches checkpoint num_classes
                img_channel=3
            )

        # Fasnet will use 2 distinct models to predict, then it will find the sum of predictions
        # to make a final prediction

        first_model = FasNetBackbone.MiniFASNetV2(conv6_kernel=(5, 5)).to(device)
        second_model = create_custom_minifasnet_se().to(device)

        # load model weight for first model
        state_dict = torch.load(first_model_weight_file, map_location=device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()

        if first_layer_name.find("module.") >= 0:
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            first_model.load_state_dict(new_state_dict)
        else:
            first_model.load_state_dict(state_dict)

        # load model weight for second model
        state_dict = torch.load(second_model_weight_file, map_location=device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()

        if first_layer_name.find("module.model.") >= 0:
            # Handle case where keys have "module.model." prefix
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                # Only process keys that start with "module.model." and ignore others like "module.FTGenerator."
                if key.startswith("module.model."):
                    name_key = key[13:]  # Remove "module.model." prefix (13 characters)
                    new_state_dict[name_key] = value
            second_model.load_state_dict(new_state_dict)
        elif first_layer_name.find("module.") >= 0:
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            second_model.load_state_dict(new_state_dict)
        elif first_layer_name.find("model.") >= 0:
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                # Only process keys that start with "model." and ignore others like "FTGenerator."
                if key.startswith("model."):
                    name_key = key[6:]  # Remove "model." prefix (6 characters)
                    new_state_dict[name_key] = value
            second_model.load_state_dict(new_state_dict)
        else:
            second_model.load_state_dict(state_dict)

        # evaluate models
        _ = first_model.eval()
        _ = second_model.eval()

        self.first_model = first_model
        self.second_model = second_model

    def analyze(self, img: np.ndarray, facial_area: Union[list, tuple]):
        """
        Analyze a given image spoofed or not
        Args:
            img (np.ndarray): pre loaded image
            facial_area (list or tuple): facial rectangle area coordinates with x, y, w, h respectively
        Returns:
            result (tuple): a result tuple consisting of is_real and score
        """
        import torch
        import torch.nn.functional as F

        x, y, w, h = facial_area
        first_img = crop(img, (x, y, w, h), 2.7, 80, 80)
        # Use larger input size for the second model to accommodate 8x8 kernels
        second_img = crop(img, (x, y, w, h), 4, 128, 128)

        test_transform_first = Compose([ToTensor()])  # First model: no normalization
        test_transform_second = Compose([ToTensorNormalized()])  # Second model: with normalization

        first_img = test_transform_first(first_img)
        first_img = first_img.unsqueeze(0).to(self.device)

        second_img = test_transform_second(second_img)
        second_img = second_img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            first_result = self.first_model.forward(first_img)
            first_result = F.softmax(first_result).cpu().numpy()

            # Debug the second model forward pass
            print(f"Second img shape: {second_img.shape}")
            print(f"Second img min/max: {second_img.min():.3f}/{second_img.max():.3f}")

            second_result_raw = self.second_model.forward(second_img)
            print(f"Second model raw output: {second_result_raw}")
            print(f"Second model raw output shape: {second_result_raw.shape}")

            second_result = F.softmax(second_result_raw).cpu().numpy()
            print(f"Second model after softmax: {second_result}")

        prediction = np.zeros((1, 3))
        # prediction += first_result
        # prediction += second_result

        label = np.argmax(prediction)
        is_real = True if label == 1 else False  # pylint: disable=simplifiable-if-expression
        score = prediction[0][label] / 2
        print(f"Fasnet spoofing prediction: label={label} is_real={is_real}, score={score}, first_result={first_result}, second_result={second_result}")
        print(f"- First model prediction: {first_result}")
        print(f"-- Is Spoof......: {first_result[0][0]}")
        print(f"-- Is Real.......: {first_result[0][1]}")
        print(f"-- Is Uncertain..: {first_result[0][2]}")
        print(f"- Second model prediction: {second_result}")
        # print(f"-- Is Spoof......: {second_result[0][0]}")
        # print(f"-- Is Real.......: {second_result[0][1]}")
        # print(f"-- Is Uncertain..: {second_result[0][2]}")

        return is_real, score


# subsdiary classes and functions


def to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    import torch

    # handle numpy array
    # IR image channel=1: modify by lzc --> 20190730
    if pic.ndim == 2:
        pic = pic.reshape((pic.shape[0], pic.shape[1], 1))

    img = torch.from_numpy(pic.transpose((2, 0, 1)))
    # backward compatibility
    # return img.float().div(255)  modify by zkx
    return img.float()


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ToTensor:
    def __call__(self, pic):
        return to_tensor(pic)


class ToTensorNormalized:
    def __call__(self, pic):
        """Convert to tensor and normalize to [0, 1] range"""
        import torch
        
        # handle numpy array
        if pic.ndim == 2:
            pic = pic.reshape((pic.shape[0], pic.shape[1], 1))

        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        return img.float().div(255)  # Normalize to [0, 1]


def _get_new_box(src_w, src_h, bbox, scale):
    x = bbox[0]
    y = bbox[1]
    box_w = bbox[2]
    box_h = bbox[3]
    scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
    new_width = box_w * scale
    new_height = box_h * scale
    center_x, center_y = box_w / 2 + x, box_h / 2 + y
    left_top_x = center_x - new_width / 2
    left_top_y = center_y - new_height / 2
    right_bottom_x = center_x + new_width / 2
    right_bottom_y = center_y + new_height / 2
    if left_top_x < 0:
        right_bottom_x -= left_top_x
        left_top_x = 0
    if left_top_y < 0:
        right_bottom_y -= left_top_y
        left_top_y = 0
    if right_bottom_x > src_w - 1:
        left_top_x -= right_bottom_x - src_w + 1
        right_bottom_x = src_w - 1
    if right_bottom_y > src_h - 1:
        left_top_y -= right_bottom_y - src_h + 1
        right_bottom_y = src_h - 1
    return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)


def crop(org_img, bbox, scale, out_w, out_h):
    src_h, src_w, _ = np.shape(org_img)
    left_top_x, left_top_y, right_bottom_x, right_bottom_y = _get_new_box(src_w, src_h, bbox, scale)
    img = org_img[left_top_y : right_bottom_y + 1, left_top_x : right_bottom_x + 1]
    dst_img = cv2.resize(img, (out_w, out_h))
    return dst_img
