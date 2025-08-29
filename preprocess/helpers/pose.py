import numpy as np
import supervision as sv

OP19_SKELETON_1B = [
    (1,2),
    (2,3),(3,4),(4,5),          # right arm
    (2,6),(6,7),(7,8),          # left arm
    (2,9),(9,10),(10,11),(11,12),  # spine → right leg
    (9,13),(13,14),(14,15),        # spine → left leg
    (1,16),(16,18),              # face right: Nose–REye–REar
    (1,17),(17,19),              # face left:  Nose–LEye–LEar
]


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes from [x, y, w, h] to [x1, y1, x2, y2].
    
    Args:
        boxes (np.ndarray): shape (N, 4), each row is [x, y, w, h]

    Returns:
        np.ndarray: shape (N, 4), each row is [x1, y1, x2, y2]
    """
    boxes = boxes.astype(np.float32)
    xyxy = boxes.copy()
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x + w
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y + h
    return xyxy




def op25_to_op19(keypoints: np.ndarray) -> np.ndarray:
    """
    Reduce OpenPose Body-25 keypoints to Body-19 by dropping foot points.
    Supports shapes:
      (25,2), (P,25,2), (25,3), (P,25,3)
    Returns the same shape with 25→19 in the joint dimension.
    """
    OP25_TO_OP19_IDX = np.arange(19, dtype=int)  # [0, 1, ..., 18]
    kp = np.asarray(keypoints)
    if kp.ndim < 2 or kp.shape[-2] != 25:
        raise ValueError(f"Expected ...x25xC, got {kp.shape}")
    return kp[..., OP25_TO_OP19_IDX, :]