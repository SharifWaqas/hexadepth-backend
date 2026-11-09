# Models (not in git)

| Detector | File | Where it is looked up |
|---|---|---|
| `authors_yolov4` | `yolov4-custom_5000.weights` + `yolov4-custom.txt` (the cfg) | `data/real/milco/Training/` (ships with the MILCO dataset) |
| `yolo` | `yolo_milco_v8n.pt` | this folder |

If the `MODEL_DIR` environment variable is set, it is the only place searched, for both models.
A missing model never crashes anything: the evaluation and the website fall back to the next
detector (`authors_yolov4` -> `yolo` -> `classical`) and say so. See `docs/model_card.md`.
