import numpy as np
import cv2


# scores: shape - (n,), dtype - float
# objects_ids (classes_ids or tracking_ids): shape - (n,), dtype - int
# boxes: shape - (n, 4), [x1, y1, x2, y2], dtype - int
# masks: shape - (n, h, w), dtype - np.uint8
def draw_objects(image, scores, objects_ids, boxes=None, masks=None, min_score=0.0,
        draw_scores=False, draw_ids=False, draw_boxes=False, draw_masks=False,
        palette=((0, 0, 255),), color_by_object_id=False):
    if boxes is None and masks is None:
        raise RuntimeError("Both boxes and masks are None")

    if boxes is None:
        boxes = [None] * len(scores)
    if masks is None:
        masks = [None] * len(scores)

    overlay = image.copy()
    for i, (score, object_id, box, mask) in enumerate(zip(scores, objects_ids, boxes, masks)):
        if score < min_score:
            continue

        if color_by_object_id:
            color = palette[object_id % len(palette)]
        else:
            color = palette[i % len(palette)]

        if draw_scores or draw_ids:
            text = list()
            if draw_scores:
                text.append(f"{score:.02f}")
            if draw_ids:
                text.append(f"id: {object_id}")
            text = ", ".join(text)
            if box is not None:
                x, y = box[0], box[1]
            else:
                nonzero_y, nonzero_x = np.nonzero(mask)
                x, y = nonzero_x.min(), nonzero_y.min()
            y -= 5
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            cv2.putText(image, text, (x, y), font, font_scale, color, thickness=2)

        if draw_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)

        if draw_masks:
            overlay[mask != 0] = np.array(color, dtype=np.uint8)
            if not draw_boxes:
                polygons, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.polylines(image, polygons, True, color, thickness=2)

    cv2.addWeighted(image, 0.7, overlay, 0.3, 0, dst=image)
