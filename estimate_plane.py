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


def estimate_plane(points_file):
    points = np.load(points_file)
    n = points.shape[0]

    # ax + by + c = z

    # Ap = B
    # p = [a, b, c]
    # A = [[xi, yi, 1]]
    # B = [zi]

    # p = inv(AT * A) * AT * B

    A = np.hstack((points[:, 0:2], np.ones((n, 1))))
    B = points[:, 2]
    p = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), B)
    print(p)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    estimate_plane(args.points_file)
