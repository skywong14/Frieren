# Frieren HD Single Full Dataset Review

- Total merged images: 80
- Dataset root: `data/datasets/frieren_hd_single80_v1`
- Source breakdown: {"single": 30, "single2": 50}
- Caption breakdown: {"legacy_manual": 30, "heuristic_auto": 50}
- Intended use: main single-character training set for the full LoRA experiment

Notes:
- Images are normalized to RGB PNG with deterministic names.
- The original 30 Single captions are reused from the manually refined set.
- The new 50 JPG images use reproducible heuristic captions and can be refined later without changing the file naming scheme.
- Diffusers compatibility requirement: keep only `images/` and `metadata.jsonl` under the dataset root.
