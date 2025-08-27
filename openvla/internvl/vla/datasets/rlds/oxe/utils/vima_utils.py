import math
import tensorflow as tf

def quaternion_to_rpy(quaternion):
    x, y, z, w = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = tf.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = tf.where(
        tf.abs(sinp) >= 1,
        tf.sign(sinp) * tf.constant(math.pi / 2),
        tf.asin(sinp)
    )

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = tf.atan2(siny_cosp, cosy_cosp)

    rpy = tf.stack([roll, pitch, yaw], axis=-1)

    return rpy

def rpy_to_quaternion(roll, pitch, yaw):
    cy = tf.cos(yaw * 0.5)
    sy = tf.sin(yaw * 0.5)
    cp = tf.cos(pitch * 0.5)
    sp = tf.sin(pitch * 0.5)
    cr = tf.cos(roll * 0.5)
    sr = tf.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return tf.stack([w, x, y, z], axis=0)
