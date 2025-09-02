import joblib
from pathlib import Path

import numpy as np

def load_default_camdicts(phalp_res_path):
    """
    camdict is dictionary composed of
    - fid
    - w2c
    - projection
    - f
    - cx
    - cy
    - H
    - W
    which all python native variable or numpy array
    TODO: taken from gtu, give them cred
    """
    phalp_res = joblib.load(phalp_res_path)

    cam_dicts = dict()
    for k in sorted(list(phalp_res.keys())):
        v = phalp_res[k]
        if len(v['size']) == 0:
            print(f"skipping detction due to non valid detection")
            continue

        H = v['size'][0][0]
        W = v['size'][0][1]
        cam_dict = dict()
        f = 5000 / 256 * max(H, W)                                                # PHALP default settings
        cx = W / 2
        cy = H / 2
        w2c = np.eye(4, dtype=np.float32)
        intrinsic = np.array([
            [f, 0., cx, 0.],
            [0., f, cy, 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])
        fid = int(Path(k).name.split(".")[0])
        
        cam_dict['fid'] = fid
        cam_dict['H'] = H
        cam_dict['W'] = W
        cam_dict['f'] = f
        cam_dict['fx'] = f
        cam_dict['fy'] = f
        cam_dict['cx'] = cx
        cam_dict['cy'] = cy
        cam_dict['w2c'] = w2c
        cam_dict['intrinsic'] = intrinsic
        cam_dict['projection'] = intrinsic @ w2c
        cam_dicts[fid] = cam_dict
    
    return cam_dicts