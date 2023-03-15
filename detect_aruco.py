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


def detect_images(images_files, K, D, aruco_size, extract_all_corners,
        out_file, vis_folder=None):
    if len(osp.dirname(out_file)) != 0:
        os.makedirs(osp.dirname(out_file), exist_ok=True)
    if args.vis_folder is not None:
        os.makedirs(args.vis_folder, exist_ok=True)

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000)
    aruco_params = cv2.aruco.DetectorParameters_create()
    marker_corners_all = list()
    for image_file in images_files:
        image = cv2.imread(image_file)

        corners, ids, _ = \
            cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)
        corners = np.array(corners)
        # corners.shape = (n, 1, 4, 2)
        # ids.shape = (n, 1)
        ind = np.argsort(ids, axis=0)
        ids = np.take_along_axis(ids, ind, axis=0)
        corners = np.take_along_axis(corners, np.expand_dims(ind, axis=(-1, -2)), axis=0)

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, aruco_size, K, D)
        # rvecs.shape = (n, 1, 3)
        # tvecs.shape = (n, 1, 3)

        n = corners.shape[0]

        marker_poses = np.tile(np.eye(4), (n, 1, 1))
        for i in range(n):
            marker_poses[i, 0:3, 0:3], _ = cv2.Rodrigues(rvecs[i])
            marker_poses[i, 0:3, 3] = tvecs[i, 0]
        # marker_poses.shape = (n, 4, 4)

        corners_in_marker_frames = list()
        for i in range(n):
            corners_in_single_marker_frame = list()
            for sx, sy in [(-1, 1), (1, 1), (1, -1), (-1, -1)]:
                # top left corner first
                single_corner_in_marker_frame = np.array(
                    [aruco_size / 2 * sx,
                     aruco_size / 2 * sy,
                     0, 1]).reshape(-1, 1)
                corners_in_single_marker_frame.append(single_corner_in_marker_frame)
                if not extract_all_corners:
                    break
            corners_in_single_marker_frame = np.array(corners_in_single_marker_frame)
            corners_in_marker_frames.append(corners_in_single_marker_frame)
        corners_in_marker_frames = np.array(corners_in_marker_frames).swapaxes(0, 1)
        # corners_in_marker_frames.shape = (1 or 4, n, 4, 1)

        marker_corners = np.matmul(marker_poses, corners_in_marker_frames)
        marker_corners = marker_corners[:, :, 0:3, 0].reshape(-1, 3)
        # marker_corners.shape = (n or n * 4, 3)

        marker_corners_all.append(marker_corners)

        if vis_folder is not None:
            vis_image = image.copy()
            cv2.aruco.drawDetectedMarkers(vis_image, corners)
            cv2.drawFrameAxes(vis_image, K, D, rvecs, tvecs, aruco_size / 2)

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

    detect_images(images_files, K, D, args.aruco_size, args.extract_all_corners,
        args.out_file, vis_folder=args.vis_folder)
