import json
import os
import cv2
from PIL import Image
import numpy as np

class Avatar:

    def __init__(self):
        self.ims = []
        self.data = {}

    def load(self, path):
        with open(os.path.join(path, 'data.json'), 'r') as f:
            self.data = json.load(f)
        
        # Load images
        for i, v in enumerate(self.data['ims']):
            self.ims.append(Image.open(os.path.join(path, v['src'])))

    def render(self, frame_im, shape):

        def data2vec(data):
            if data:
                return (data['factor']*(shape[data['end']]-shape[data['start']])).astype(int)
            else:
                return np.array([0, 0])

        # Transform
        mod_ims = []
        for i, v in enumerate(self.data['ims']):

            input_im = self.ims[i]

            pa = np.float32([
                [0, 0],
                [input_im.size[0], 0],
                [input_im.size[0], input_im.size[1]],
                [0, input_im.size[1]],
            ])

            u_diff = data2vec(v['u'])
            v_diff = data2vec(v['v'])

            # Inertia, I still have no idea what this is called
            u_inertia = v['u']['inertia'] if 'inertia' in v['u'] else 0
            v_inertia = v['v']['inertia'] if 'inertia' in v['v'] else 0
            u_diff = ((1-u_inertia)*u_diff).astype(int) + (u_inertia*np.array([np.linalg.norm(u_diff), 0])).astype(int)
            v_diff = ((1-v_inertia)*v_diff).astype(int) + (v_inertia*np.array([0, np.linalg.norm(v_diff)])).astype(int)

            p_0 = shape[v['p_0']['p']] + data2vec(v['p_0']['offset_u']) + data2vec(v['p_0']['offset_v'])

            pb = np.float32([
                p_0 - u_diff - v_diff,
                p_0 + u_diff - v_diff,
                p_0 + u_diff + v_diff,
                p_0 - u_diff + v_diff,
            ])

            coeffs = cv2.getPerspectiveTransform(pb, pa).flatten()
            mod_ims.append(input_im.transform(frame_im.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC))

        # Paste
        for im in mod_ims:
            frame_im.paste(im, (0, 0), im)
        
        return frame_im