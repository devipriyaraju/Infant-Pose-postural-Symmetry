from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

LIMB_DRAW = {
    0: [(5, 7), (6, 8)],
    1: [(7, 9), (8, 10)],
    2: [(11, 13), (12, 14)],
    3: [(13, 15), (14, 16)],
}


def draw_symmetry(
    image_path,
    keypoints,
    symmetry,
    bbox=None,
    title=None,
    output_path=None,
):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(Image.open(image_path).convert("RGB"))
    ax.axis("off")

    for a, b in [(5, 6), (11, 12), (5, 11), (6, 12)]:
        ax.plot(
            *keypoints[[a, b]].T,
            color="white",
            linewidth=1.5,
        )

    for limb, segments in LIMB_DRAW.items():
        for a, b in segments:
            ax.plot(
                *keypoints[[a, b]].T,
                color="limegreen" if symmetry[limb] else "red",
                linewidth=3,
            )

    if bbox is not None:
        x0, y0, width, height = bbox
        ax.set_xlim(x0, x0 + width)
        ax.set_ylim(y0 + height, y0)

    if title:
        ax.set_title(title)

    fig.tight_layout()
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
    return fig, ax
