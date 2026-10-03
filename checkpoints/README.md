# Checkpoints

Pretrained and fine tuned checkpoint files are excluded from Git.

MMPose COCO checkpoints can be downloaded with `mim download`.

Fine tuned checkpoints from Compute Canada can be placed under:

```text
work_dirs/
├── HRNet-W32_fold0/
├── HRNet-W32_fold1/
├── ...
└── RTMPose-S_fold4/
```

The repository `.gitignore` excludes these large model files.
