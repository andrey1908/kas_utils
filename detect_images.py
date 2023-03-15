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
    parser.add_argument('-out', '--out-file', required=True, type=str)
    parser.add_argument('-vis-fld', '--vis-folder', type=str)
    return parser


def detect_images(images_files, K, D, aruco_size, out_file, vis_folder=None):
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    if args.vis_folder is not None:
        os.makedirs(args.vis_folder, exist_ok=True)

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000)
    aruco_params = cv2.aruco.DetectorParameters_create()
    tl_corners = list()
    for image_file in images_files:
        image = cv2.imread(image_file)
        corners, ids, _ = \
            cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)
        assert len(corners) == 1
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, aruco_size, K, D)

        marker_pose = np.eye(4)
        marker_pose[0:3, 0:3] = cv2.Rodrigues(rvecs)[0]
        marker_pose[0:3, 3] = tvecs[0, 0]
        tl_corner_in_marker_frame = np.array(
            [-aruco_size / 2,
             aruco_size / 2,
             0,
             1])
        tl_corner = np.matmul(marker_pose, tl_corner_in_marker_frame)
        tl_corners.append(tl_corner[0:3])

        if vis_folder is not None:
            vis_image = image.copy()
            cv2.aruco.drawDetectedMarkers(vis_image, corners)
            cv2.drawFrameAxes(vis_image, K, D, rvecs, tvecs, aruco_size / 2)
            
            vis_image_file = osp.join(
                vis_folder, Path(image_file).stem + '_vis.jpg')
            cv2.imwrite(vis_image_file, vis_image)
    tl_corners = np.array(tl_corners)
    np.save(out_file, tl_corners)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    images_files = glob.glob(args.images_folder + f"/*.{args.images_extension}")
    images_files = sorted(images_files)
    assert len(images_files) != 0

    camera_calibration = np.load(args.camera_calibration)
    K = camera_calibration['K']
    D = camera_calibration['D']

    detect_images(images_files, K, D, args.aruco_size, args.out_file,
        vis_folder=args.vis_folder)
