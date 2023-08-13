import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_mask_in_roi(mask, x_range, y_range):
    mask_in_roi = np.zeros_like(mask)
    mask_in_roi[y_range, x_range] = mask[y_range, x_range]
    return mask_in_roi


def refine_mask_by_polygons(mask, min_polygon_length=0, max_polygon_length=-1,
        min_polygon_area_length_ratio=0, select_top_n_polygons_by_length=-1):
    polygons, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    accepted_polygons = list()
    for polygon in polygons:
        if len(polygon) >= min_polygon_length and \
                (max_polygon_length < 0 or len(polygon) <= max_polygon_length):
            if min_polygon_area_length_ratio == 0 or \
                    cv2.contourArea(polygon) / len(polygon) >= min_polygon_area_length_ratio:
                accepted_polygons.append(polygon)

    if select_top_n_polygons_by_length >= 0 and \
            len(accepted_polygons) > select_top_n_polygons_by_length:
        accepted_polygons = sorted(accepted_polygons, key=lambda p: -len(p))
        accepted_polygons = accepted_polygons[:select_top_n_polygons_by_length]

    refined_mask = np.zeros_like(mask)
    for accepted_polygon in accepted_polygons:
        cv2.fillPoly(refined_mask, [accepted_polygon], 255)

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
        min_sv=0, max_sv=255,
        shift_h=0, inverse_mask=False, show_image=True):
    if select_image_roi:
        x_range, y_range = select_roi(image, full_by_default=True)
    else:
        x_range, y_range = slice(0, None), slice(0, None)
    image = image[y_range, x_range].copy()

    if min_sv != 0 or max_sv != 255:
        use_sv_limits = True
    else:
        use_sv_limits = False
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    hsv[:, :, 0] += shift_h
    if use_sv_limits:
        h = hsv[:, :, 0]
        sv = get_sv(hsv)
        h_sv = np.dstack((h, sv))
        mask = cv2.inRange(h_sv, (min_h, min_sv), (max_h, max_sv))
    else:
        mask = cv2.inRange(hsv, (min_h, min_s, min_v), (max_h, max_s, max_v))

    if not inverse_mask:
        background = (mask == 0)
    else:
        background = (mask != 0)
    image[background] = 0
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
    plt.hist(s, bins=256, range=(0, 255))
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
    plt.hist(v, bins=256, range=(0, 255))
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
    plt.hist(sv, bins=256, range=(0, 255))
    if show:
        plt.show()


def plot_h_histogram(image, select_image_roi=True,
        min_s=0, max_s=255, min_v=0, max_v=255,
        min_sv=0, max_sv=255, shift_h=0, show=True):
    if select_image_roi:
        x_range, y_range = select_roi(image, full_by_default=True)
    else:
        x_range, y_range = slice(0, None), slice(0, None)
    image = image[y_range, x_range]

    if min_sv != 0 or max_sv != 255:
        use_sv_limits = True
    else:
        use_sv_limits = False
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    if use_sv_limits:
        sv = get_sv(hsv)
        mask = cv2.inRange(sv, min_sv, max_sv)
    else:
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        s_v = np.dstack((s, v))
        mask = cv2.inRange(s_v, (min_s, min_v), (max_s, max_v))

    h = hsv[:, :, 0]
    h = h[mask != 0]
    h += shift_h
    plt.hist(h, bins=256, range=(0, 255))
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


def show_h(image, shift_h=0):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    h = hsv[:, :, 0]
    h += shift_h
    show(h)


def get_sv(hsv):
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    sv = s.astype(np.float32) * v.astype(np.float32) / 255
    sv = sv.astype(np.uint8)
    return sv


def plot_sv_points(image, select_image_roi=True,
        min_h=0, max_h=255, shift_h=0, show=True):
    if select_image_roi:
        x_range, y_range = select_roi(image, full_by_default=True)
    else:
        x_range, y_range = slice(0, None), slice(0, None)
    image = image[y_range, x_range]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    h = hsv[:, :, 0]
    h += shift_h
    mask = cv2.inRange(h, min_h, max_h)
    hsv = hsv[mask != 0]
    s = hsv[:, 1]
    v = hsv[:, 2]
    plt.xlim([0, 255])
    plt.xlabel('s')
    plt.ylim([0, 255])
    plt.ylabel('v')
    plt.plot(s, v, 'o', markersize=1)
    if show:
        plt.show()

def show_hsv_color(h, s, v):
    hsv = np.zeros((200, 200, 3), dtype=np.uint8)
    hsv[:, :, 0] = h
    hsv[:, :, 1] = s
    hsv[:, :, 2] = v
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR_FULL)
    show(bgr)
