import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_mask_in_roi(mask, x_range, y_range):
    mask_in_roi = np.zeros_like(mask)
    mask_in_roi[y_range, x_range] = mask[y_range, x_range]
    return mask_in_roi


def refine_mask_by_polygons(mask, min_polygon_length=100, max_polygon_length=1000):
    polygons, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    refined_mask = np.zeros_like(mask)
    accepted_polygons = list()
    for polygon in polygons:
        if min_polygon_length <= len(polygon) <= max_polygon_length:
            cv2.fillPoly(refined_mask, [polygon], 255)
            accepted_polygons.append(polygon)
    return refined_mask, accepted_polygons


#############


def show(image, window_name="show image"):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def select_roi(image, full_by_default=False, window_name="select roi"):
    roi = cv2.selectROI(window_name, image, showCrosshair=False)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    if (x, y, w, h) == (0, 0, 0, 0):
        if full_by_default:
            x_range = slice(0, None)
            y_range = slice(0, None)
        else:
            x_range = None
            y_range = None
    else:
        x_range = slice(x, x + w)
        y_range = slice(y, y + h)
    return x_range, y_range


def get_and_apply_mask(image, select_image_roi=True,
        min_h=0, max_h=255, min_s=0, max_s=255, min_v=0, max_v=255,
        inverse_mask=False, show_image=True):
    if select_image_roi:
        x_range, y_range = select_roi(image, full_by_default=True)
    else:
        x_range, y_range = slice(0, None), slice(0, None)
    image = image[y_range, x_range].copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    mask = cv2.inRange(hsv, (min_h, min_s, min_v), (max_h, max_s, max_v))
    if not inverse_mask:
        background_mask = (mask == 0)
    else:
        background_mask = (mask != 0)
    image[background_mask] = np.array([0, 0, 0])
    if show_image:
        show(image)
    return image, mask


def plot_s_histogram(image, select_image_roi=True, show=True):
    if select_image_roi:
        x_range, y_range = select_roi(image, full_by_default=True)
    else:
        x_range, y_range = slice(0, None), slice(0, None)
    image = image[y_range, x_range]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    s = hsv[:, :, 1]
    s = s.flatten()
    plt.hist(s, bins=256)
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
    plt.hist(v, bins=256)
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
    plt.hist(sv, bins=256)
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
    plt.hist(h, bins=256)
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
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    sv = s.astype(np.float32) * v.astype(np.float32) / 255
    sv = sv.astype(np.uint8)
    return sv


def get_mask_for_h(hsv,
        min_s=0, max_s=255, min_v=0, max_v=255,
        min_sv=0, max_sv=255):
    if min_sv != 0 or max_sv != 255:
        use_sv_limits = True
    else:
        use_sv_limits = False

    if not use_sv_limits:
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        mask = (s >= min_s) & (s <= max_s) & (v >= min_v) & (v <= max_v)
    else:
        sv = get_sv(hsv)
        mask = (sv >= min_sv) & (sv <= max_sv)
    return mask
