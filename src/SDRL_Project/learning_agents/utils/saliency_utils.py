import numpy as np


def binarize_attention_mask(attention_mask, threshold=0.1):
    """ The attention mask should be a 2d numpy array """
    attention_mask[attention_mask > threshold] = 1.0
    attention_mask[attention_mask <= threshold] = 0
    return attention_mask


def apply_mask_to_rgb(mask, rgb):
    """ apply 2d mask to a 3-channel rgb image """
    mask = mask.astype(np.float32)
    rgb = rgb.astype(np.float32)
    rgb *= mask[..., None]
    return rgb
