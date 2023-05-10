import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import show, select_roi


def segment_by_color(image, min_color, max_color, \
        x_range=slice(0, None), y_range=slice(0, None),
        refine=False, min_polygon_length=100, max_polygon_length=1000):
    mask_full = cv2.inRange(image, min_color, max_color)
    mask = np.zeros(mask_full.shape, dtype=mask_full.dtype)
    mask[y_range, x_range] = mask_full[y_range, x_range]
    if refine:
        polygons, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        refined_mask = np.zeros(mask.shape, dtype=mask.dtype)
        accepted_polygons = list()
        for polygon in polygons:
            if min_polygon_length <= len(polygon) <= max_polygon_length:
                cv2.fillPoly(refined_mask, [polygon], 1)
                accepted_polygons.append(polygon)
        return refined_mask, accepted_polygons
    else:
        return mask


def plot_s_histogram(image, select_image_roi=True, show=True):
    if select_image_roi:
        x_range, y_range = select_roi(image, full_by_default=True)
    else:
        x_range, y_range = slice(0, None), slice(0, None)
    image = image[y_range, x_range]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    s = hsv[:, :, 1]
    s = s.flatten()
    plt.hist(s, bins=255)
    if show:
        plt.show()


def plot_v_histogram(image, select_image_roi=True, show=True):
    if select_image_roi:
        x_range, y_range = select_roi(image, full_by_default=True)
    else:
        x_range, y_range = slice(0, None), slice(0, None)
    image = image[y_range, x_range]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    v = hsv[:, :, 2]
    v = v.flatten()
    plt.hist(v, bins=255)
    if show:
        plt.show()


def plot_sv_histogram(image, select_image_roi=True, show=True):
    if select_image_roi:
        x_range, y_range = select_roi(image, full_by_default=True)
    else:
        x_range, y_range = slice(0, None), slice(0, None)
    image = image[y_range, x_range]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    sv = get_sv(hsv)
    sv = sv.flatten()
    plt.hist(sv, bins=255)
    if show:
        plt.show()


def plot_h_histogram(image, select_image_roi=True,
        min_s=0, max_s=255, min_v=0, max_v=255,
        min_sv=0, max_sv=255, show=True):
    if select_image_roi:
        x_range, y_range = select_roi(image)
    else:
        x_range, y_range = slice(0, None), slice(0, None)
    image = image[y_range, x_range]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    h = hsv[:, :, 0]

    mask = get_mask_for_h(hsv, min_s, max_s, min_v, max_v, min_sv, max_sv)
    h = h[mask]
    plt.hist(h, bins=255)
    if show:
        plt.show()


def show_s(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    s = hsv[:, :, 1]
    show(s)


def show_v(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    v = hsv[:, :, 2]
    show(v)


def show_sv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    sv = get_sv(hsv)
    show(sv)


def show_h(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    h = hsv[:, :, 0]
    show(h)


def get_sv(hsv):
    _, s, v = np.split(hsv, 3, axis=-1)
    sv = s.astype(np.float32) * v.astype(np.float32) / 255
    sv = sv.astype(np.uint8)
    return sv


def get_mask_for_h(hsv,
        min_s, max_s, min_v, max_v,
        min_sv, max_sv):
    if min_s == 0 and max_s == 255 and min_v == 0 and max_v == 255:
        use_sv_limits = True
    else:
        use_sv_limits = False

    if not use_sv_limits:
        _, s, v = np.split(hsv, 3, axis=-1)
        mask = (s >= min_s) & (s <= max_s) & (v >= min_v) & (v <= max_v)
    else:
        sv = get_sv(hsv)
        mask = (sv >= min_sv) & (sv <= max_sv)
    return mask
