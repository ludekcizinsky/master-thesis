import joblib
from pathlib import Path
import json

import numpy as np
from preprocess.helpers.video_utils import extract_frame_id

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
        fid = extract_frame_id(Path(k).name)

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

def save_camdicts_json(cam_dicts, save_path):
    serializable = {}
    for fid, cam in cam_dicts.items():
        serializable[fid] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in cam.items()
        }
    with open(save_path, "w") as f:
        json.dump(serializable, f)
    print(f"Saved cam_dicts to {save_path}")

def load_camdicts_json(load_path):
    with open(load_path, "r") as f:
        data = json.load(f)
    cam_dicts = {}
    for fid, cam in data.items():
        cam_dicts[int(fid)] = {
            k: np.array(v, dtype=np.float32) if isinstance(v, list) else v
            for k, v in cam.items()
        }
    print(f"Loaded cam_dicts from {load_path}, {len(cam_dicts)} frames")
    return cam_dicts