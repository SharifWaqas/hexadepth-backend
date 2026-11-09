# Original sonar image examples used by HexaDepth

**Source:** https://huggingface.co/datasets/rehan9599/drishti-sss  
**License:** CC BY-SA 4.0  

- `view_samples.html`: open this in a web browser to see **actual images hosted on Hugging Face** and open each original file. Requires internet access.
- `download_samples.py`: run `python download_samples.py` on your internet-connected computer to download all 8 original image files **and their corresponding YOLO label `.txt` files** to `downloaded_sonar_images/`. Python standard library only. The ZIP does **not** contain image binaries until you run the downloader.

The current checkpointed training script selected `.jpg` files from the `train` split. The seven real images here (.jpg) therefore belong to its selected training image set. The `synth_ghost_net_00001.png` sample is included to demonstrate a **synthetic dataset category** but **was not included in that JPG-only training run**. The DRISHTI release has no labeled `crab_pot` examples.

## Attribution

- **DRISHTI:** Rehan Fazal and collaborators, [Sonar-Drishti](https://github.com/Rehan9599/Sonar-Drishti). Combined dataset is [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- Pipeline and background tiles: Álvarez-Tuñón et al., [SubPipe / SubPipeMini2](https://zenodo.org/records/10808161), CC BY 4.0.
- Shipwreck tiles: Sethuraman et al., [AI4Shipwrecks](https://deepblue.lib.umich.edu/data/concern/data_sets/8623hz41x), CC BY 4.0.
- Cylindrical-object tiles: [Sonar Imaging Mine Detection](https://www.kaggle.com/datasets/sierra022/sonar-imaging-mine-detection), CC BY-SA 4.0.
- Synthetic ghost-net sample: DRISHTI's procedural acoustic generator, CC BY-SA 4.0.

Use these training-split examples only for **design and illustration** or training inspection, not as independent held-out evaluation data. A deployed model's actual predictions may differ from ground-truth labels.

## Dashboard bundle
Only the seven real JPG images are bundled here, unchanged. The synthetic ghost-net image is excluded. Original YOLO annotations, if present, are reference data and are not used as model predictions.
