# Copyright (c) Megvii Inc. All rights reserved.

import random

import cv2
import numpy as np

__all__ = ["visualize_assign"]


def random_color():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)


def visualize_assign(img, boxes, coords, match_results, save_name=None) -> np.ndarray:
    """visualize label assign result.

    Args:
        img: img to visualize
        boxes: gt boxes in xyxy format
        coords: coords of matched anchors
        match_results: match results of each gt box and coord.
        save_name: name of save image, if None, image will not be saved. Default: None.
    """
    for box_id, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = random_color()
        assign_coords = coords[match_results == box_id]
        if assign_coords.numel() == 0:
            # unmatched boxes are red
            color = (0, 0, 255)
            cv2.putText(
                img, "unmatched", (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1
            )
        else:
            for coord in assign_coords:
                # draw assigned anchor
                cv2.circle(img, (int(coord[0]), int(coord[1])), 3, color, -1)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    if save_name is not None:
        cv2.imwrite(save_name, img)

    return img
