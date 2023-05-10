import cv2


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