# Frieren HD Dataset V1 Review

- Total raw images: 50
- Single images: 30
- Multi images: 20
- Recommended core training set: `data/datasets/frieren_hd_single_v1`
- Recommended use of multi images: hold out for later mixed-scene experiments after manual review

Notes:
- All files were normalized to RGB PNG and renamed deterministically.
- Captions are first-pass structured captions generated from folder semantics and image orientation.
- These captions are good enough for a first training pass, but manual refinement is still recommended before a final report run.
