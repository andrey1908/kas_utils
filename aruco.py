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


def detect_aruco(image, K=None, D=None, aruco_sizes=None, use_generic=False,
        aruco_dict=cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000),
        params=cv2.aruco.DetectorParameters_create()):
    corners, ids, rejected = \
        cv2.aruco.detectMarkers(image, aruco_dict, parameters=params)
    n = len(corners)
    n_rejected = len(rejected)

    if not use_generic:
        n_poses = 1
    else:
        n_poses = 2

    if n != 0:
        corners = np.array(corners)
        ids = np.array(ids)
        # corners.shape = (n, 1, 4, 2)
        # ids.shape = (n, 1)

        ind = np.argsort(ids, axis=0)
        ids = np.take_along_axis(ids, ind, axis=0)
        corners = np.take_along_axis(corners, np.expand_dims(ind, axis=(-1, -2)), axis=0)

        # estimate 3d poses
        if all(item is not None for item in (K, D, aruco_sizes)):
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
                aruco_size = aruco_sizes[i]
                obj = np.array([
                    [-aruco_size / 2,  aruco_size / 2, 0],
                    [ aruco_size / 2,  aruco_size / 2, 0],
                    [ aruco_size / 2, -aruco_size / 2, 0],
                    [-aruco_size / 2, -aruco_size / 2, 0]])
                if not use_generic:
                    retval, rvec, tvec = \
                        cv2.solvePnP(obj, corners[i], K, D,
                            flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    rvec = rvec.swapaxes(0, 1)
                    tvec = tvec.swapaxes(0, 1)
                    # rvec.shape = (1, 3)
                    # tvec.shape = (1, 3)
                else:
                    retval, rvec, tvec, reprojectionError = \
                        cv2.solvePnPGeneric(obj, corners[i], K, D,
                            flags=cv2.SOLVEPNP_IPPE_SQUARE,
                            reprojectionError=np.empty(0, dtype=np.float))
                    assert len(rvec) == n_poses
                    assert reprojectionError[0][0] <= reprojectionError[1][0]
                    rvec = np.array(rvec)
                    tvec = np.array(tvec)
                    # rvec.shape = (2, 3, 1)
                    # tvec.shape = (2, 3, 1)

                    rvec = rvec.squeeze()
                    tvec = tvec.squeeze()
                    # rvec.shape = (2, 3)
                    # tvec.shape = (2, 3)
                rvecs.append(rvec)
                tvecs.append(tvec)
            rvecs = np.array(rvecs)
            tvecs = np.array(tvecs)
            # rvecs.shape = (n, n_poses, 3)
            # tvecs.shape = (n, n_poses, 3)

            marker_poses = np.tile(np.eye(4), (n, n_poses, 1, 1))
            for i in range(n):
                for j in range(n_poses):
                    marker_poses[i, j, 0:3, 0:3], _ = cv2.Rodrigues(rvecs[i, j])
                    marker_poses[i, j, 0:3, 3] = tvecs[i, j]
            # marker_poses.shape = (n, n_poses, 4, 4)

            corners_3d_in_marker_frames = list()
            for i in range(n):
                corners_3d_in_single_marker_frame = list()
                aruco_size = aruco_sizes[i]
                for sx, sy in [(-1, 1), (1, 1), (1, -1), (-1, -1)]:
                    single_corner_3d_in_marker_frame = np.array([
                        aruco_size / 2 * sx,
                        aruco_size / 2 * sy,
                        0, 1]).reshape(4, 1)
                    corners_3d_in_single_marker_frame.append(single_corner_3d_in_marker_frame)
                corners_3d_in_single_marker_frame = np.array(corners_3d_in_single_marker_frame)
                corners_3d_in_marker_frames.append(corners_3d_in_single_marker_frame)
            corners_3d_in_marker_frames = np.array(corners_3d_in_marker_frames)
            # corners_3d_in_marker_frames.shape = (n, 4, 4, 1)

            corners_3d_in_marker_frames = \
                np.expand_dims(corners_3d_in_marker_frames.swapaxes(0, 1), axis=2)
            # corners_3d_in_marker_frames.shape = (4, n, 1, 4, 1)

            corners_3d = np.matmul(marker_poses, corners_3d_in_marker_frames)
            corners_3d = corners_3d[:, :, :, 0:3, 0].transpose(1, 2, 0, 3)
            # corners_3d.shape = (n, n_poses, 4, 3)
        else:
            n_poses = 0
            aruco_sizes = None
            rvecs = None
            tvecs = None
            corners_3d = None
    else:
        corners = np.empty((0, 1, 4, 2))
        ids = np.empty((0, 1))
        aruco_sizes = np.empty((0,))
        rvecs = np.empty((0, n_poses, 3))
        tvecs = np.empty((0, n_poses, 3))
        corners_3d = np.empty((0, n_poses, 4, 3))

    if n_rejected != 0:
        rejected = np.array(rejected)
        # rejected.shape = (n_rejected, 1, 4, 2)
    else:
        rejected = np.empty((0, 1, 4, 2))

    # corners.shape = (n, 1, 4, 2)
    # ids.shape = (n, 1)
    # rejected.shape = (n_rejected, 1, 4, 2)
    # aruco_sizes.shape = (n,)
    # rvecs.shape = (n, n_poses, 3)
    # tvecs.shape = (n, n_poses, 3)
    # corners_3d.shape = (n, n_poses, 4, 3)
    return {
        'corners': corners, 'ids': ids, 'n': n,
        'rejected': rejected, 'n_rejected': n_rejected,
        'aruco_sizes': aruco_sizes,
        'rvecs': rvecs, 'tvecs': tvecs, 'n_poses': n_poses,
        'corners_3d': corners_3d}


def draw_aruco(image, arucos, draw_ids=False, K=None, D=None):
    if draw_ids:
        cv2.aruco.drawDetectedMarkers(image, arucos['corners'], arucos['ids'])
    else:
        cv2.aruco.drawDetectedMarkers(image, arucos['corners'])
    if all(item is not None for item in (arucos['aruco_sizes'], K, D)):
        for i in range(arucos['n']):
            cv2.drawFrameAxes(image, K, D,
                arucos['rvecs'][i], arucos['tvecs'][i], arucos['aruco_sizes'][i] / 2)


def detect_aruco_common(images_files, K, D, aruco_size,
        out_file, vis_folder=None):
    if len(osp.dirname(out_file)) != 0:
        os.makedirs(osp.dirname(out_file), exist_ok=True)
    if vis_folder is not None:
        os.makedirs(vis_folder, exist_ok=True)

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000)
    aruco_params = cv2.aruco.DetectorParameters_create()
    # aruco_params.adaptiveThreshConstant = 14
    corners_3d_all = list()
    for image_file in images_files:
        image = cv2.imread(image_file)
        arucos = detect_aruco(image, K, D, aruco_size,
            aruco_dict, aruco_params)
        corners_3d_all.append(arucos['corners_3d'])

        n = arucos['n']
        print(f"{image_file} : detected {n} marker{'' if n == 1 else 's'}")

        if vis_folder is not None:
            corners = arucos['corners']
            rvecs = arucos['rvecs']
            tvecs = arucos['tvecs']
            vis_image = image.copy()
            draw_aruco(vis_image, corners, K=K, D=D,
                rvecs=rvecs, tvecs=tvecs, frames_sizes=aruco_size / 2)
            vis_image_file = osp.join(
                vis_folder, Path(image_file).stem + '_vis.jpg')
            cv2.imwrite(vis_image_file, vis_image)

    corners_3d_all = np.vstack(corners_3d_all)
    np.save(out_file, corners_3d_all)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    images_files = glob.glob(args.images_folder + f"/*.{args.images_extension}")
    images_files = sorted(images_files)
    assert len(images_files) != 0

    camera_calibration = np.load(args.camera_calibration)
    K = camera_calibration['K']
    D = camera_calibration['D']

    detect_aruco_common(images_files, K, D, args.aruco_size,
        args.out_file, vis_folder=args.vis_folder)
