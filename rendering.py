import cv2
from PIL import Image
import numpy as np

import utils

shades_im = Image.open('shades.png')
face_im = Image.open('miho_head.png')
hair_im = Image.open('miho_hair.png')
eye_im = Image.open('miho_eye.png')


def transform_shades(input_im, frame_im, shape):
    """Transforms shades image
    """

    pa = [
        [0, 0],
        [input_im.size[0], 0],
        [input_im.size[0], input_im.size[1]],
        [0, input_im.size[1]],
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

    coeffs = utils.find_coeffs(pb, pa)
    return input_im.transform(frame_im.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def transform_face(input_im, frame_im, shape):
    """Transforms face
    """

    pa = [
        [0, 0],
        [input_im.size[0], 0],
        [input_im.size[0], face_im.size[1]],
        [0, input_im.size[1]],
    ]

    u_diff = (0.8*(shape[16]-shape[0])).astype(int)
    v_diff = (0.8*(shape[8]-shape[27])).astype(int)

    pb = [
        shape[30] - u_diff - v_diff,
        shape[30] + u_diff - v_diff,
        shape[30] + u_diff + v_diff,
        shape[30] - u_diff + v_diff,
    ]
    
    coeffs = utils.find_coeffs(pb, pa)
    return input_im.transform(frame_im.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)

def transform_hair(input_im, frame_im, shape):

    pa = [
        [0, 0],
        [input_im.size[0], 0],
        [input_im.size[0], input_im.size[1]],
        [0, input_im.size[1]],
    ]

    u_diff = (1.2*(shape[16]-shape[0])).astype(int)
    v_diff = (1.2*(shape[8]-shape[27])).astype(int)

    pb = [
        shape[30] - u_diff - v_diff,
        shape[30] + u_diff - v_diff,
        shape[30] + u_diff + v_diff,
        shape[30] - u_diff + v_diff,
    ]
    
    coeffs = utils.find_coeffs(pb, pa)
    return input_im.transform(frame_im.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)

def transform_left_eye(input_im, frame_im, shape):

    pa = [
        [0, 0],
        [input_im.size[0], 0],
        [input_im.size[0], input_im.size[1]],
        [0, input_im.size[1]],
    ]

    u_diff = (shape[45]-shape[42]).astype(int)
    v_diff = (3*(shape[46]-shape[44])).astype(int)

    u_diff_1 = (0.3*(shape[16]-shape[0])).astype(int)
    v_diff_1 = (0.1*(shape[8]-shape[27])).astype(int)
    p_0 = u_diff_1 + v_diff_1

    pb = [
        shape[30] - u_diff - v_diff + p_0,
        shape[30] + u_diff - v_diff + p_0,
        shape[30] + u_diff + v_diff + p_0,
        shape[30] - u_diff + v_diff + p_0,
    ]
    
    coeffs = utils.find_coeffs(pb, pa)
    return input_im.transform(frame_im.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)

def transform_right_eye(input_im, frame_im, shape):

    pa = [
        [0, 0],
        [input_im.size[0], 0],
        [input_im.size[0], input_im.size[1]],
        [0, input_im.size[1]],
    ]

    u_diff = (shape[39]-shape[36]).astype(int)
    v_diff = (3*(shape[41]-shape[37])).astype(int)

    u_diff_1 = (0.3*(shape[16]-shape[0])).astype(int)
    v_diff_1 = (0.1*(shape[8]-shape[27])).astype(int)
    p_0 = -u_diff_1 + v_diff_1

    pb = [
        shape[30] - u_diff - v_diff + p_0,
        shape[30] + u_diff - v_diff + p_0,
        shape[30] + u_diff + v_diff + p_0,
        shape[30] - u_diff + v_diff + p_0,
    ]
    
    coeffs = utils.find_coeffs(pb, pa)
    return input_im.transform(frame_im.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def render(frame_im, shape):
    """Render images
    """

    shades_im_mod = transform_shades(shades_im, frame_im, shape)
    face_im_mod = transform_face(face_im, frame_im, shape)
    hair_im_mod = transform_hair(hair_im, frame_im, shape)
    left_eye_im_mod = transform_left_eye(eye_im, frame_im, shape)
    right_eye_im_mod = transform_right_eye(eye_im.transpose(Image.FLIP_LEFT_RIGHT), frame_im, shape)

    # Paste
    frame_im.paste(face_im_mod, (0, 0), face_im_mod)
    frame_im.paste(hair_im_mod, (0, 0), hair_im_mod)
    frame_im.paste(left_eye_im_mod, (0, 0), left_eye_im_mod)
    frame_im.paste(right_eye_im_mod, (0, 0), right_eye_im_mod)
    #frame_im.paste(shades_im_mod, (0, 0), shades_im_mod)

    return frame_im
