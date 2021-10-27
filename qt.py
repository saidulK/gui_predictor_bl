

import numpy as np
import math
class Quaternion(object):
    """A class for describing a quaternion."""

    def __init__(self, q0=0, q1=0, q2=0, q3=0):
        self.q0 = q0
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.values = [q0, q1, q2, q3]


def quaternion_product(p, q):
    """p, q are two quaternions; quaternion product."""

    p0 = p.q0
    p1 = p.q1
    p2 = p.q2
    p3 = p.q3

    q0 = q.q0
    q1 = q.q1
    q2 = q.q2
    q3 = q.q3

    r0 = p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3
    r1 = p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2
    r2 = p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1
    r3 = p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0

    r = Quaternion(r0, r1, r2, r3)

    return (r)


def rotate_vector(q, v):
    """q is the quaternion describing the rotation to apply, v is the vector on
    which to apply the rotation"""

    quaternion_v = Quaternion(0, v[0], v[1], v[2])
    transposed_q = Quaternion(q.q0, -q.q1, -q.q2, -q.q3)

    r = quaternion_product(quaternion_product(q, quaternion_v), transposed_q)
    return [r.q1, r.q2, r.q3]


def gyro_to_quaternion(w, dt):
    """w is the vector indicating angular rate in the reference frame of the
    IMU, all coords in rad/s
    dt is the time interval during which the angular rate is valid"""

    wx = w[0]
    wy = w[1]
    wz = w[2]

    l = (wx ** 2 + wy ** 2 + wz ** 2) ** 0.5

    dtlo2 = dt * l / 2

    q0 = np.cos(dtlo2)
    q1 = np.sin(dtlo2) * wx / l
    q2 = np.sin(dtlo2) * wy / l
    q3 = np.sin(dtlo2) * wz / l

    r = Quaternion(q0, q1, q2, q3)

    return (r)

def mg(v):
    sum=0
    for i in range(len(v)):
        sum+=(v[i]**2)


    return (sum)**0.5


def acc_to_quat(acc, total, act_g, g=None):
    
    if g is None:
        X, Y, Z = total[0] - acc[0], total[1] - acc[1], total[2] - acc[2]
    else:
        X, Y, Z = g[0] ,g[1], g[2]
        
    gs = np.array([X, Y, Z])
    gs = gs / mg(gs)
    gr = np.array(act_g)
    gr = gr/mg(gr)

    v = np.cross(gs, gr)

    if np.dot(gs, gr) < -0.999999:
        xarray = np.array([1, 0, 0])
        yarray = np.array([0, 1, 0])
        tmp = np.cross(gs, xarray)

        if mg(tmp) < 0.000001:
            tmp = np.cross(gs, yarray)

        v = np.copy(tmp)
        v = v / mg(v)
        alpha = math.pi

    else:
        if mg(v):
            v = v / mg(v)
        alpha = np.arccos(np.dot(gs, gr) / (mg(gs) * mg(gr)))

    q0 = np.cos(alpha / 2)
    q1 = v[0] * np.sin(alpha / 2)
    q2 = v[1] * np.sin(alpha / 2)
    q3 = v[2] * np.sin(alpha / 2)

    q = Quaternion(q0, q1, q2, q3)
    mag = (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) ** 0.5

    q0 = q0 / mag
    q1 = q1 / mag
    q2 = q2 / mag
    q3 = q3 / mag

    q = Quaternion(q0, q1, q2, q3)

    return (q)