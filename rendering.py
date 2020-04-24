import cv2
from PIL import Image
import numpy as np

shades_im = Image.open('shades.png')
face_im = Image.open('miho_head.png')
hair_im = Image.open('miho_hair.png')
left_eye_im = Image.open('miho_left_eye.png')
right_eye_im = Image.open('miho_right_eye.png')

def transform_shades(input_im, frame_im, shape):
    """Transforms shades image
    """

    pa = np.float32([
        [0, 0],
        [input_im.size[0], 0],
        [input_im.size[0], input_im.size[1]],
        [0, input_im.size[1]],
    ])

    u_diff = (1.5*(shape[45]-shape[36])).astype(int)
    v_diff = (1.5*(shape[30]-shape[27])).astype(int)
    v_0 = (0.3*(shape[30]-shape[27])).astype(int)

    pb = np.float32([
        shape[27] - u_diff - v_diff - v_0,
        shape[27] + u_diff - v_diff - v_0,
        shape[27] + u_diff + v_diff - v_0,
        shape[27] - u_diff + v_diff - v_0,
    ])

    coeffs = cv2.getPerspectiveTransform(pb, pa).flatten()
    return input_im.transform(frame_im.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def transform_face(input_im, frame_im, shape):
    """Transforms face
    """

    pa = np.float32([
        [0, 0],
        [input_im.size[0], 0],
        [input_im.size[0], face_im.size[1]],
        [0, input_im.size[1]],
    ])

    u_diff = (0.7*(shape[16]-shape[0])).astype(int)
    v_diff = (0.8*(shape[8]-shape[27])).astype(int)

    pb = np.float32([
        shape[30] - u_diff - v_diff,
        shape[30] + u_diff - v_diff,
        shape[30] + u_diff + v_diff,
        shape[30] - u_diff + v_diff,
    ])
    
    coeffs = cv2.getPerspectiveTransform(pb, pa).flatten()
    return input_im.transform(frame_im.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)

def transform_hair(input_im, frame_im, shape):

    pa = np.float32([
        [0, 0],
        [input_im.size[0], 0],
        [input_im.size[0], input_im.size[1]],
        [0, input_im.size[1]],
    ])

    u_diff = (1.1*(shape[16]-shape[0])).astype(int)
    v_diff = (1.2*(shape[8]-shape[27])).astype(int)

    pb = np.float32([
        shape[30] - u_diff - v_diff,
        shape[30] + u_diff - v_diff,
        shape[30] + u_diff + v_diff,
        shape[30] - u_diff + v_diff,
    ])
    
    coeffs = cv2.getPerspectiveTransform(pb, pa).flatten()
    return input_im.transform(frame_im.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)

def transform_left_eye(input_im, frame_im, shape):

    pa = np.float32([
        [0, 0],
        [input_im.size[0], 0],
        [input_im.size[0], input_im.size[1]],
        [0, input_im.size[1]],
    ])

    u_diff = (shape[45]-shape[42]).astype(int)
    v_diff = (4*(shape[46]-shape[44])).astype(int)

    u_0 = (0.28*(shape[16]-shape[0])).astype(int)
    v_0 = (0.12*(shape[8]-shape[27])).astype(int)
    p_0 = u_0 + v_0

    pb = np.float32([
        shape[30] - u_diff - v_diff + p_0,
        shape[30] + u_diff - v_diff + p_0,
        shape[30] + u_diff + v_diff + p_0,
        shape[30] - u_diff + v_diff + p_0,
    ])
    
    coeffs = cv2.getPerspectiveTransform(pb, pa).flatten()
    return input_im.transform(frame_im.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)

def transform_right_eye(input_im, frame_im, shape):

    pa = np.float32([
        [0, 0],
        [input_im.size[0], 0],
        [input_im.size[0], input_im.size[1]],
        [0, input_im.size[1]],
    ])

    u_diff = (shape[39]-shape[36]).astype(int)
    v_diff = (4*(shape[41]-shape[37])).astype(int)

    u_0 = (0.28*(shape[16]-shape[0])).astype(int)
    v_0 = (0.12*(shape[8]-shape[27])).astype(int)
    p_0 = -u_0 + v_0

    pb = np.float32([
        shape[30] - u_diff - v_diff + p_0,
        shape[30] + u_diff - v_diff + p_0,
        shape[30] + u_diff + v_diff + p_0,
        shape[30] - u_diff + v_diff + p_0,
    ])
    
    coeffs = cv2.getPerspectiveTransform(pb, pa).flatten()
    return input_im.transform(frame_im.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def render(frame_im, shape):
    """Render images
    """

    shades_im_mod = transform_shades(shades_im, frame_im, shape)
    face_im_mod = transform_face(face_im, frame_im, shape)
    hair_im_mod = transform_hair(hair_im, frame_im, shape)
    left_eye_im_mod = transform_left_eye(left_eye_im, frame_im, shape)
    right_eye_im_mod = transform_right_eye(right_eye_im, frame_im, shape)

    # Paste
    frame_im.paste(face_im_mod, (0, 0), face_im_mod)
    frame_im.paste(hair_im_mod, (0, 0), hair_im_mod)
    frame_im.paste(left_eye_im_mod, (0, 0), left_eye_im_mod)
    frame_im.paste(right_eye_im_mod, (0, 0), right_eye_im_mod)
    
    #frame_im.paste(shades_im_mod, (0, 0), shades_im_mod)

    return frame_im
