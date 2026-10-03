import cv2
import numpy as np


CORRUPTIONS = [
    "clean",
    "occlude_limb",
    "dark",
    "bright",
    "blur",
    "noise",
    "rotate15",
]


def corrupt(image, kind, keypoints, bbox_xywh, rng):
    height, width = image.shape[:2]
    x0, y0, box_width, box_height = bbox_xywh

    output = image.copy()
    keypoints_out = keypoints.copy()
    box = np.array(
        [x0, y0, x0 + box_width, y0 + box_height],
        dtype=float,
    )
    target = -1

    if kind == "clean":
        return output, keypoints_out, box, target

    if kind == "occlude_limb":
        candidates = [
            joint
            for joint in (7, 8, 9, 10, 13, 14, 15, 16)
            if keypoints[joint, 2] > 0
        ]
        target = int(rng.choice(candidates))
        size = int(0.25 * max(box_width, box_height))
        center_x, center_y = keypoints[target, :2].astype(int)
        mean_value = image.reshape(-1, 3).mean(0)
        output[
            max(0, center_y - size // 2): center_y + size // 2,
            max(0, center_x - size // 2): center_x + size // 2,
        ] = mean_value

    elif kind == "dark":
        output = np.clip(
            image.astype(float) * 0.35,
            0,
            255,
        ).astype(np.uint8)

    elif kind == "bright":
        output = np.clip(
            image.astype(float) * 1.6 + 40,
            0,
            255,
        ).astype(np.uint8)

    elif kind == "blur":
        kernel = max(3, int(0.03 * max(box_width, box_height)) | 1)
        output = cv2.GaussianBlur(image, (kernel, kernel), 0)

    elif kind == "noise":
        output = np.clip(
            image.astype(float)
            + rng.normal(0, 25, image.shape),
            0,
            255,
        ).astype(np.uint8)

    elif kind == "rotate15":
        center = (
            x0 + box_width / 2,
            y0 + box_height / 2,
        )
        matrix = cv2.getRotationMatrix2D(center, 15, 1.0)
        output = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            borderMode=cv2.BORDER_CONSTANT,
        )

        homogeneous = np.c_[
            keypoints[:, :2],
            np.ones(17),
        ]
        keypoints_out[:, :2] = homogeneous @ matrix.T

        corners = np.array(
            [
                [box[0], box[1], 1],
                [box[2], box[1], 1],
                [box[0], box[3], 1],
                [box[2], box[3], 1],
            ]
        ) @ matrix.T
        box = np.r_[
            corners.min(0),
            corners.max(0),
        ]
        box = np.clip(
            box,
            0,
            [width - 1, height - 1, width - 1, height - 1],
        )
    else:
        raise ValueError(f"Unknown corruption: {kind}")

    return output, keypoints_out, box, target
