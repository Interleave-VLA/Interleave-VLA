import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

def qmul(pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
    p1, q1 = pose1[..., : 3], R.from_quat(pose1[..., 3 : 7])
    p2, q2 = pose2[..., : 3], R.from_quat(pose2[..., 3 : 7])
    new_q = q1 * q2 # quaternion_multiply(pose.q, arg0.q)
    new_p = p1 + q1.apply(p2) # pose.p + quaternion_apply(pose.q, arg0.p)
    return np.concatenate([new_p, new_q.as_quat()], axis=-1)

def qdiv(pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
    p1, q1 = pose1[..., : 3], R.from_quat(pose1[..., 3 : 7])
    p2, q2 = pose2[..., : 3], R.from_quat(pose2[..., 3 : 7])
    new_q = q1 * q2.inv()
    new_p = p1 - new_q.apply(p2)
    return np.concatenate([new_p, new_q.as_quat()], axis=-1)

def resize(img_array: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    img = Image.fromarray(img_array)
    # img.thumbnail(target_size, Image.Resampling.LANCZOS) # same aspect ratio
    img = img.resize(target_size) # different aspect ratio
    # img.save("resized.jpg")
    return np.array(img)

if __name__ == '__main__':
    pose1 = np.random.rand(7)
    pose1[3 : 7] /= np.linalg.norm(pose1[3 : 7])
    pose1 = np.tile(pose1, (10, 1))
    pose2 = np.random.rand(7)
    pose2[3 : 7] /= np.linalg.norm(pose2[3 : 7])
    pose2 = np.tile(pose2, (10, 1))
    print(qmul(qdiv(pose1, pose2), pose2))
    print(pose1)
    print(qdiv(qmul(pose1, pose2), pose2))
    print(pose1)