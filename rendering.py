import cv2
from PIL import Image
import numpy as np

import utils

shades_im = Image.open('shades.png')

def find_shades_coeffs(frame, shape):
    """Find transformation coefficients for shades
    """

    pa = [
        [0, 0],
        [shades_im.size[0], 0],
        [shades_im.size[0], shades_im.size[1]],
        [0, shades_im.size[1]],
    ]

    u_diff = (1.5*(shape[45]-shape[36])).astype(int)
    v_diff = (1.5*(shape[30]-shape[27])).astype(int)
    v_0 = (0.3*(shape[30]-shape[27])).astype(int)

    pb = [
        shape[27] - u_diff - v_diff - v_0,
        shape[27] + u_diff - v_diff - v_0,
        shape[27] + u_diff + v_diff - v_0,
        shape[27] - u_diff + v_diff - v_0,
    ]
    
    return utils.find_coeffs(pb, pa)


def render(frame, shape):
    """Render images
    """

    frame_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Paste images
    shades_im_mod = shades_im.transform(frame_im.size, Image.PERSPECTIVE, find_shades_coeffs(frame, shape), Image.BICUBIC)
    frame_im.paste(shades_im_mod, (0, 0), shades_im_mod)

    frame = cv2.cvtColor(np.array(frame_im), cv2.COLOR_RGB2BGR)

    return frame