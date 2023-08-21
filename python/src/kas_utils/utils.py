import numpy as np
import cv2
import os
import os.path as osp
import glob


def show(image, window_name="image", destroy_window=True):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    key = cv2.waitKey(0)
    if destroy_window:
        cv2.destroyWindow(window_name)
    return key


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


def get_depth_scale(depth):
    if isinstance(depth, np.ndarray):
        dtype = depth.dtype
    else:
        dtype = depth

    if dtype == float:
        scale = 1
    elif dtype == np.uint16:
        scale = 0.001
    else:
        raise RuntimeError(f"Unknown depth type {dtype}")

    return scale


class SavePathsGenerator:
    def __init__(self, save_folder, extension, continue_saving=False,
            start_from=0):
        if extension[0] != '.':
            raise RuntimeError("SavePathsGenerator: use '.' before extension.")

        self.save_folder = osp.expanduser(save_folder)
        self.extension = extension
        self.continue_saving = continue_saving
        self.start_from = start_from

        if self.continue_saving:
            files = glob.glob(f"{self.save_folder}/????{self.extension}")
            max_num = -1
            for file in files:
                num = osp.splitext(osp.basename(file))[0]
                if num.isdigit():
                    num = int(num)
                    max_num = max(max_num, num)
            self.counter = max_num + 1
        else:
            self.counter = self.start_from

    def __call__(self):
        return self.next()

    def next(self):
        next_save_path = f'{self.counter:04}{self.extension}'
        next_save_path = osp.join(self.save_folder, next_save_path)
        self.counter += 1
        return next_save_path
