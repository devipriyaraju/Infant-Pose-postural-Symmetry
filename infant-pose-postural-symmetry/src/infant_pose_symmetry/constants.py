KP_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]

FLIP_PERM = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

COCO_IDX = {
    "ls": 5, "rs": 6, "le": 7, "re": 8, "lw": 9, "rw": 10,
    "lh": 11, "rh": 12, "lk": 13, "rk": 14, "la": 15, "ra": 16,
}

LSP_IDX = {
    "ra": 0, "rk": 1, "rh": 2, "lh": 3, "lk": 4, "la": 5,
    "rw": 6, "re": 7, "rs": 8, "ls": 9, "le": 10, "lw": 11,
}

LIMBS = ["upper_arm", "lower_arm", "upper_leg", "lower_leg"]

LIMB_JOINTS = {
    "upper_arm": (5, 7, 6, 8),
    "lower_arm": (7, 9, 8, 10),
    "upper_leg": (11, 13, 12, 14),
    "lower_leg": (13, 15, 14, 16),
}

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
]

MIRROR_EDGES = [
    (1, 2), (3, 4), (5, 6), (7, 8),
    (9, 10), (11, 12), (13, 14), (15, 16),
]
