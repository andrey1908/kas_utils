import argparse
import numpy as np
import cv2
import glob
import os
import os.path as osp
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fld', '--images-folder', required=True, type=str)
    parser.add_argument('-ext', '--images-extension', type=str, default='jpg')
    parser.add_argument('-calib', '--camera-calibration', required=True, type=str)
    parser.add_argument('-size', '--aruco-size', required=True, type=float)
    parser.add_argument('-all-corns', '--extract-all-corners', action='store_true')
    parser.add_argument('-out', '--out-file', required=True, type=str)
    parser.add_argument('-vis-fld', '--vis-folder', type=str)
    return parser


def detect_aruco(image, K, D, aruco_sizes, extract_all_corners,
        aruco_dict, aruco_params):
    corners, ids, rejected = \
        cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)
    corners = np.array(corners)
    rejected = np.array(rejected)
    n = corners.shape[0]
    rejected_n = rejected.shape[0]
    # corners.shape = (n, 1, 4, 2)
    # ids.shape = (n, 1)
    # rejected.shape = (rejected_n, 1, 4, 2)

    if n != 0:
        ind = np.argsort(ids, axis=0)
        ids = np.take_along_axis(ids, ind, axis=0)
        corners = np.take_along_axis(corners, np.expand_dims(ind, axis=(-1, -2)), axis=0)

        if isinstance(aruco_sizes, (list, tuple)):
            aruco_sizes = np.array(aruco_sizes)
        elif not isinstance(aruco_sizes, np.ndarray):
            aruco_sizes = np.array([aruco_sizes] * n)
        if len(aruco_sizes.shape) != 1:
            raise RuntimeError(f"Use list, tuple or np.ndarray to pass multiple aruco sizes.")
        if aruco_sizes.shape != (n,):
            raise RuntimeError(
                f"Number of aruco marker sizes does not correspond to "
                f"the number of detected markers ({aruco_sizes.shape[0]} vs {n})")

        rvecs = list()
        tvecs = list()
        for i in range(n):
            rvec, tvec, _ = \
                cv2.aruco.estimatePoseSingleMarkers(corners[i], aruco_sizes[i], K, D)
            rvecs.append(rvec)
            tvecs.append(tvec)
        rvecs = np.array(rvecs)
        tvecs = np.array(tvecs)
        # rvecs.shape = (n, 1, 3)
        # tvecs.shape = (n, 1, 3)

        marker_poses = np.tile(np.eye(4), (n, 1, 1))
        for i in range(n):
            marker_poses[i, 0:3, 0:3], _ = cv2.Rodrigues(rvecs[i])
            marker_poses[i, 0:3, 3] = tvecs[i, 0]
        # marker_poses.shape = (n, 4, 4)

        corners_3d_in_marker_frames = list()
        for i in range(n):
            corners_3d_in_single_marker_frame = list()
            for sx, sy in [(-1, 1), (1, 1), (1, -1), (-1, -1)]:
                # top left corner first
                single_corner_3d_in_marker_frame = np.array(
                    [aruco_sizes[i] / 2 * sx,
                    aruco_sizes[i] / 2 * sy,
                    0, 1]).reshape(-1, 1)
                corners_3d_in_single_marker_frame.append(single_corner_3d_in_marker_frame)
                if not extract_all_corners:
                    break
            corners_3d_in_single_marker_frame = np.array(corners_3d_in_single_marker_frame)
            corners_3d_in_marker_frames.append(corners_3d_in_single_marker_frame)
        corners_3d_in_marker_frames = np.array(corners_3d_in_marker_frames).swapaxes(0, 1)
        # corners_3d_in_marker_frames.shape = (1 or 4, n, 4, 1)

        marker_corners_3d = np.matmul(marker_poses, corners_3d_in_marker_frames)
        marker_corners_3d = marker_corners_3d[:, :, 0:3, 0].swapaxes(0, 1).reshape(-1, 3)
        # marker_corners_3d.shape = (n or n * 4, 3)
    else:
        marker_corners_3d = np.empty((0, 3))
        rvecs = np.empty((0, 1, 3))
        tvecs = np.empty((0, 1, 3))

    return marker_corners_3d, \
        {'n': n, 'corners': corners, 'ids': ids,
        'rejected': rejected, 'rejected_n': rejected_n,
        'rvecs': rvecs, 'tvecs': tvecs}


def draw_aruco(image, corners, ids=None,
        K=None, D=None, rvecs=None, tvecs=None, frames_sizes=None):
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
    else:
        cv2.aruco.drawDetectedMarkers(image, corners)
    if all(item is not None for item in (K, D, rvecs, tvecs, frames_sizes)):
        n = rvecs.shape[0]
        if not isinstance(frames_sizes, (list, tuple, np.ndarray)):
            frames_sizes = np.array([frames_sizes] * n)
        if len(frames_sizes.shape) != 1:
            raise RuntimeError(
                f"Use list, tuple or np.ndarray to pass multiple frames sizes.")
        if tvecs.shape[0] != n or frames_sizes.shape[0] != n:
            raise RuntimeError(
                f"Num of rvecs - {n}, num of tvecs - {tvecs.shape[0]}, "
                f"num of frames_sizes - {frames_sizes.shape[0]}. "
                "All these should be equal.")
        for i in range(n):
            cv2.drawFrameAxes(image, K, D, rvecs[i], tvecs[i], frames_sizes[i])


def detect_aruco_common(images_files, K, D, aruco_size, extract_all_corners,
        out_file, vis_folder=None):
    if len(osp.dirname(out_file)) != 0:
        os.makedirs(osp.dirname(out_file), exist_ok=True)
    if vis_folder is not None:
        os.makedirs(vis_folder, exist_ok=True)

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000)
    aruco_params = cv2.aruco.DetectorParameters_create()
    # aruco_params.adaptiveThreshConstant = 14
    marker_corners_all = list()
    for image_file in images_files:
        image = cv2.imread(image_file)
        marker_corners, verb = detect_aruco(image, K, D, aruco_size,
            extract_all_corners, aruco_dict, aruco_params)
        marker_corners_all.append(marker_corners)

        n = verb['n']
        print(f"{image_file} : detected {n} marker{'' if n == 1 else 's'}")

        if vis_folder is not None:
            corners = verb['corners']
            rvecs = verb['rvecs']
            tvecs = verb['tvecs']
            vis_image = image.copy()
            draw_aruco(vis_image, corners, K=K, D=D,
                rvecs=rvecs, tvecs=tvecs, frames_sizes=aruco_size / 2)
            vis_image_file = osp.join(
                vis_folder, Path(image_file).stem + '_vis.jpg')
            cv2.imwrite(vis_image_file, vis_image)

    marker_corners_all = np.vstack(marker_corners_all)
    np.save(out_file, marker_corners_all)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    images_files = glob.glob(args.images_folder + f"/*.{args.images_extension}")
    images_files = sorted(images_files)
    assert len(images_files) != 0

    camera_calibration = np.load(args.camera_calibration)
    K = camera_calibration['K']
    D = camera_calibration['D']

    detect_aruco_common(images_files, K, D, args.aruco_size, args.extract_all_corners,
        args.out_file, vis_folder=args.vis_folder)
