# coding=utf-8
import numpy as np
import transforms3d
import math
from scipy.optimize import least_squares
import pyopengv
import argparse


# Checks if a matrix is a valid rotation matrix.
# from https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# from https://www.learnopencv.com/rotation-matrix-to-euler-angles/
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def calc_inintial_guess(target_points, source_points, method="UPNP"):
    target_bearings = target_points / np.linalg.norm(target_points, axis=1).reshape(-1, 1)
    source_bearings = source_points / np.linalg.norm(source_points, axis=1).reshape(-1, 1)
 
    if method == "RANSAC":
        transformation = \
            pyopengv.absolute_pose_ransac(target_bearings, source_bearings,
                "UPNP", 0.001, 100000)
    elif method == "EPNP":
        transformation = \
            pyopengv.absolute_pose_epnp(target_bearings, source_bearings)
    elif method == "UPNP":
        transformation = \
            pyopengv.absolute_pose_upnp(target_bearings, source_bearings)[0]
    else:
        raise Exception("Opengv method error!")

    angs = rotationMatrixToEulerAngles(transformation[:3, :3].T).tolist()
    ret = list()
    ret.extend(angs)
    ret.extend((-transformation[:3, 3]).tolist())
    return np.array(ret)


def roate_with_rt(r_t, arr):  
    rot_mat = np.dot(transforms3d.axangles.axangle2mat([0, 0, 1], r_t[2]),
                     np.dot(transforms3d.axangles.axangle2mat([0, 1, 0], r_t[1]),
                            transforms3d.axangles.axangle2mat([1, 0, 0], r_t[0])))

    transformed_pcd = np.dot(rot_mat, arr.T).T + r_t[[3, 4, 5]]
    return transformed_pcd


def cost_func(r_t, target_points, source_points):  
    transformed_points = roate_with_rt(r_t, source_points)
    residuals = np.linalg.norm(target_points - transformed_points, axis=1)
    return residuals


def run_min(args, initial_guess):
    res = least_squares(cost_func, initial_guess, args=args, ftol=1e-15, max_nfev=100000)
    return res


def opt_r_t(target_points, source_points, initial_guess):
    args = (target_points, source_points)
    res = run_min(args, initial_guess)
    return res.x


def opt(target_points, source_points):
    initial_guess = calc_inintial_guess(target_points, source_points, method="UPNP")
    print(initial_guess)
    res = opt_r_t(target_points, source_points, initial_guess)
    print(res)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tp', '--target-points', required=True, type=str)
    parser.add_argument('-sp', '--source-points', required=True, type=str)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    target_points = np.load(args.target_points)
    source_points = np.load(args.source_points)

    opt(target_points, source_points)
