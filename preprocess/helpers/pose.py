import numpy as np
import supervision as sv

OP18_SKELETON = [
    (1,2),
    (2,3),(3,4),(4,5),        # right arm
    (2,6),(6,7),(7,8),        # left arm
    (9,10),(10,11),           # right leg
    (12,13),(13,14),          # left leg
    (1,15),(15,17),           # face right: Nose->REye->REar
    (1,16),(16,18),           # face left:  Nose->LEye->LEar
]


def xywh_to_xyxy(b):
    x, y, w, h = b
    return np.array([x, y, x + w, y + h], dtype=np.float32)

def joints_to_keypoints(joints, K=18):
    # joints: list length K, each (x,y) or None -> (1,K,2) with NaNs for missing
    arr = np.full((1, K, 2), np.nan, dtype=np.float32)
    for i, p in enumerate(joints[:K]):
        if p is not None:
            arr[0, i] = p
    return sv.KeyPoints(xy=arr)  # confidence/class_id optional

def op25_to_op18(op_jnts):
    j_inds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=np.int32)

    op_joints = []
    for j_ind in j_inds:
        if j_ind >= len(op_jnts):
            op_joints.append(None)
        else:
            op_joints.append(op_jnts[j_ind])

    return op_joints