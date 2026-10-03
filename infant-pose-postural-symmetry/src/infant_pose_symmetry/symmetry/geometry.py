import numpy as np

from infant_pose_symmetry.constants import COCO_IDX, LIMB_JOINTS


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    cos = (unit(u) * unit(v)).sum(-1)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def limb_angles(points: np.ndarray, joint_index=COCO_IDX) -> np.ndarray:
    points = np.asarray(points, dtype=float)

    def reflect(v, normal):
        return v - 2 * (v * normal).sum(-1, keepdims=True) * normal

    shoulder_normal = unit(points[:, joint_index["ls"]] - points[:, joint_index["rs"]])
    hip_normal = unit(points[:, joint_index["lh"]] - points[:, joint_index["rh"]])

    upper_arm = angle(
        points[:, joint_index["le"]] - points[:, joint_index["ls"]],
        reflect(points[:, joint_index["re"]] - points[:, joint_index["rs"]], shoulder_normal),
    )
    lower_arm = angle(
        points[:, joint_index["lw"]] - points[:, joint_index["le"]],
        reflect(points[:, joint_index["rw"]] - points[:, joint_index["re"]], shoulder_normal),
    )
    upper_leg = angle(
        points[:, joint_index["lk"]] - points[:, joint_index["lh"]],
        reflect(points[:, joint_index["rk"]] - points[:, joint_index["rh"]], hip_normal),
    )
    lower_leg = angle(
        points[:, joint_index["la"]] - points[:, joint_index["lk"]],
        reflect(points[:, joint_index["ra"]] - points[:, joint_index["rk"]], hip_normal),
    )

    return np.stack([upper_arm, lower_arm, upper_leg, lower_leg], axis=1)


def angle_class(values: np.ndarray) -> np.ndarray:
    return np.digitize(values, [30, 60])


def canonicalize(keypoints: np.ndarray):
    keypoints = np.asarray(keypoints, dtype=float)
    shoulder = (keypoints[:, 5] + keypoints[:, 6]) / 2
    hip = (keypoints[:, 11] + keypoints[:, 12]) / 2
    center = (shoulder + hip) / 2
    torso = shoulder - hip

    scale = np.linalg.norm(torso, axis=1)
    span = np.linalg.norm(keypoints.max(1) - keypoints.min(1), axis=1)
    scale = np.where(scale < 0.05 * span, span / 3, scale) + 1e-6

    theta = np.arctan2(torso[:, 0], -torso[:, 1])
    rotation = np.stack(
        [
            np.stack([np.cos(theta), np.sin(theta)], axis=1),
            np.stack([-np.sin(theta), np.cos(theta)], axis=1),
        ],
        axis=1,
    )
    normalized = np.einsum(
        "nij,nkj->nki",
        rotation,
        keypoints - center[:, None],
    ) / scale[:, None, None]
    return normalized, theta, scale


def segment_confidence(scores: np.ndarray) -> np.ndarray:
    output = np.zeros((len(scores), 4, 2), dtype=float)
    for limb_index, joints in enumerate(LIMB_JOINTS.values()):
        left_a, left_b, right_a, right_b = joints
        output[:, limb_index, 0] = (scores[:, left_a] + scores[:, left_b]) / 2
        output[:, limb_index, 1] = (scores[:, right_a] + scores[:, right_b]) / 2
    return output


def geometric_features(keypoints: np.ndarray, scores: np.ndarray):
    normalized, theta, scale = canonicalize(keypoints)
    angles = limb_angles(keypoints)

    def segment_length(a, b):
        return np.linalg.norm(keypoints[:, a] - keypoints[:, b], axis=1) + 1e-6

    length_ratio = np.stack(
        [
            np.log(segment_length(a, b) / segment_length(c, d))
            for a, b, c, d in LIMB_JOINTS.values()
        ],
        axis=1,
    )
    confidence = segment_confidence(scores).min(-1)
    torso_twist = angle(keypoints[:, 5] - keypoints[:, 6], keypoints[:, 11] - keypoints[:, 12])
    widths = np.stack([segment_length(5, 6), segment_length(11, 12)], axis=1)
    widths = widths / scale[:, None]

    image_features = np.c_[
        angles / 90,
        length_ratio,
        torso_twist / 90,
        np.sin(theta),
        np.cos(theta),
        widths,
        normalized.reshape(len(normalized), -1),
        scores,
    ]
    limb_features = np.stack([angles / 90, length_ratio, confidence], axis=-1)
    return image_features, limb_features


def row_features(keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray:
    image_features, limb_features = geometric_features(keypoints, scores)
    limb_identity = np.tile(np.eye(4), (len(keypoints), 1))
    return np.c_[
        limb_identity,
        limb_features.reshape(-1, 3),
        np.repeat(image_features, 4, axis=0),
    ]
