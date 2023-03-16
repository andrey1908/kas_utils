import argparse
import numpy as np
import cv2
import glob
import os
import os.path as osp
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--points-file', required=True, type=str)
    return parser


def project_to_plane(p0, plane):
    x0 = p0[0]
    y0 = p0[1]
    z0 = p0[2]
    a = plane[0]
    b = plane[1]
    c = plane[2]

    dz = (a * x0 + b * y0 + c - z0) / (a * a + b * b + 1)
    x = x0 - a * dz
    y = y0 - b * dz
    z = z0 + dz

    return np.array([x, y, z])


def estimate_plane(points_file):
    points = np.load(points_file)
    n = points.shape[0]

    # ax + by + c = z

    # A * plane = B
    # plane = [a, b, c]
    # A = [[xi, yi, 1]]
    # B = [zi]

    # plane = inv(AT * A) * AT * B

    A = np.hstack((points[:, 0:2], np.ones((n, 1))))
    B = points[:, 2]
    plane = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), B)

    centroid = np.sum(points, axis=0) / n
    origin = project_to_plane(centroid, plane)

    a = plane[0]
    b = plane[1]
    c = plane[2]
    z_axis = np.array([-a, -b, 1])
    z_axis /= np.linalg.norm(z_axis)
    x_axis = project_to_plane(points[0], plane) - origin
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    if np.dot(z_axis, origin) > 0:
        y_axis *= -1
        z_axis *= -1

    x_axis = np.expand_dims(x_axis, axis=-1)
    y_axis = np.expand_dims(y_axis, axis=-1)
    z_axis = np.expand_dims(z_axis, axis=-1)

    R = np.hstack((x_axis, y_axis, z_axis))
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = origin
    return T


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    T = estimate_plane(args.points_file)
    print(T)
