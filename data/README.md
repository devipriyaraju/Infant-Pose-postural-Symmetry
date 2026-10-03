# Data directory

Dataset images are intentionally excluded from Git.

Recommended layout:

```text
data/
├── raw/
│   └── expanded_SyRIP/
├── processed/
├── annotations/
└── splits/
```

The preparation code recursively discovers COCO keypoint annotation JSON files and the associated images.

Do not commit SyRIP images unless the dataset terms explicitly allow redistribution.
