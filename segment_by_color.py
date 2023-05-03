import numpy as np
import cv2


def segment_by_color(image, min_color, max_color, \
        x_range=slice(0, None), y_range=slice(0, None),
        refine_mask=False, min_polygon_length=100, max_polygon_length=1000):
    mask_full = cv2.inRange(image, min_color, max_color)
    mask = np.zeros(mask_full.shape, dtype=mask_full.dtype)
    mask[y_range, x_range] = mask_full[y_range, x_range]
    if refine_mask:
        polygons, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        refined_mask = np.zeros(mask.shape, dtype=mask.dtype)
        num = 0
        out_polygons = list()
        for polygon in polygons:
            if min_polygon_length <= len(polygon) <= max_polygon_length:
                cv2.fillPoly(refined_mask, [polygon], 1)
                out_polygons.append(polygon)
                num += 1
        return refined_mask, num, out_polygons
    else:
        return mask