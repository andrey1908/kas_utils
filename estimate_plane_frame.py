import argparse
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--points-file', required=True, type=str)
    return parser


def project_to_XY(points, T):
    assert points.shape[-1] == 3
    orig_shape = points.shape
    points = points.reshape(-1, 3)

    n = points.shape[0]
    points = np.hstack((points, np.ones((n, 1))))
    points = np.expand_dims(points, axis=-1)
    # points.shape = (n, 4, 1)

    points_in_T = np.matmul(np.linalg.inv(T), points)
    points_in_T[:, 2, 0] = 0
    projected_points = np.matmul(T, points_in_T)
    # projected_points.shape = (n, 4, 1)

    projected_points = projected_points[:, 0:3, 0].reshape(*orig_shape)
    return projected_points


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


def estimate_plane_frame(points):
    # ax + by + c = z

    # A * plane = B
    # plane = [a, b, c]
    # A = [[xi, yi, 1]]
    # B = [zi]

    # plane = inv(AT * A) * AT * B

    n = points.shape[0]
    assert n % 2 == 0

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

    x_direction = \
        (np.sum(points[:int(n/2)], axis=0) / (n/2) +
        2 * centroid - np.sum(points[int(n/2):], axis=0) / (n/2)) / 2
    x_axis = project_to_plane(x_direction, plane) - origin
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


def estimate_plane_frame_common(points_file):
    points = np.load(points_file)
    T = estimate_plane_frame(points)
    return T


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    T = estimate_plane_frame_common(args.points_file)
    print(T)
