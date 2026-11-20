# Findings

Every number here comes from a JSON file in `reports/`, and the source is named in brackets. The full tables are in [results.md](results.md). Explanations marked **Hypothesis** are plausible reasons that I have **not** tested. They should not be read as facts.

## 1. In-distribution performance (CIFAKE test, 20,000 images)

| | SmallCNN | ResNet18 |
|---|---|---|
| Accuracy | 96.5% | 97.8% |
| ROC-AUC | 0.9943 | 0.9977 |
| Parameters | 355,873 | 11,177,025 |
| CPU time per image | 0.23 ms | 3.14 ms |

[`metrics_smallcnn.json`, `metrics_resnet18.json`]

- **Size vs gain:** ResNet18 has 31x more parameters and is about 14x slower per image on CPU, but gains only 1.3 accuracy points. Both models are very strong **on CIFAKE**.
- **Errors are balanced** for both models. SmallCNN made 390 false positives and 311 false negatives. ResNet18 made 259 and 185.

## 2. Calibration

[`calibration_*.json`]

- **Already calibrated:** on CIFAKE both original models were already well calibrated. SmallCNN's ECE was 0.0049 and ResNet18's was 0.0028. The fitted temperatures were close to 1 (0.952 and 1.029), so temperature scaling changed little: ECE went to 0.0041 and 0.0025.
- **Fine-tuned models:** these were mis-calibrated on CIFAKE after fine-tuning, and temperature scaling clearly helped:
  - SmallCNN_mix: ECE 0.0373 to 0.0086, with T = 0.620. It was *under*-confident, so its probabilities were sharpened.
  - ResNet18_mix: ECE 0.0147 to 0.0033, with T = 1.501. It was *over*-confident, so its probabilities were softened.
- **Caveat:** T is fitted on CIFAKE validation data only. We did **not** measure calibration on COCOAI or on real uploads, so the app's probabilities are only known to be calibrated for CIFAKE-like images.

## 3. Robustness to everyday image processing

[`robustness_smallcnn.json`, `robustness_resnet18.json`]

| Degradation | SmallCNN accuracy | ResNet18 accuracy | Change from clean (ResNet18) |
|---|---|---|---|
| none | 96.5% | 97.8% | |
| JPEG q90 | 96.4% | 97.8% | −0.0 pts |
| JPEG q70 | 95.8% | 97.5% | −0.3 pts |
| JPEG q50 | 93.0% | 96.0% | −1.8 pts |
| JPEG q30 | 91.3% | 91.9% | −5.9 pts |
| resize to 75% and back | 64.9% | 64.5% | **−33.3 pts** |
| resize to 50% and back | 68.7% | 62.6% | **−35.2 pts** |
| blur radius 0.5 | 81.2% | 87.1% | −10.7 pts |
| blur radius 1.0 | 65.6% | 59.3% | **−38.5 pts** |

- **JPEG:** both models are fairly robust down to quality 50. Note that every CIFAKE file is *already* a JPEG at estimated quality 75 [`jpeg_quality.json`], so the models were trained on JPEG artefacts in both classes.
- **Resizing and blur are far more damaging than JPEG.** The drop comes almost entirely from **missed fakes**. ResNet18's recall falls from 98.2% to 29.4% after the 75% resize and to 20.1% after a radius-1.0 blur. ROC-AUC falls less (to 0.908 and 0.790), so part of the problem is the threshold shifting rather than the ranking disappearing completely.
  - **Hypothesis:** the models detect fine, high-frequency patterns left by Stable Diffusion 1.4's decoder. Blurring and resampling remove exactly those patterns, so a fake starts to look "real". Real CIFAR-10 photos were themselves heavily downsampled and are smooth.
- **Unexpected result:** SmallCNN did slightly *better* at a 50% resize than at 75% (68.7% vs 64.9%).
  - **Hypothesis:** at 75% the bilinear resampling grid produces interpolation patterns that are not aligned with the pixel grid, and these confuse the model. We did not investigate this.
- **Practical meaning:** messaging apps resize and recompress images, so the in-app result for a shared image is much less reliable than the CIFAKE accuracy suggests. The app shows warnings for this.

## 4. Cross-generator generalisation (COCOAI)

[`cocoai_*.json`, `cocoai_subset.json`]

**The test set:** 1,000 real COCO photos and 1,000 images from each of SD 2.1, SDXL, SD 3, DALL·E 3 and Midjourney 6. The split is made by caption, so no scene appears both here and in the fine-tuning data.

**Models trained only on CIFAKE fail badly:**

| | SmallCNN | ResNet18 |
|---|---|---|
| COCOAI ROC-AUC (all generators) | 0.581 | 0.610 |
| Real photos correctly called real | 72.9% | **7.7%** |
| Real vs DALL·E 3, AUC | 0.433 | 0.520 |
| Real vs Midjourney 6, AUC | 0.429 | 0.437 |
| Real vs SD 2.1, AUC | 0.804 | 0.753 |

- **Overall drop:** ROC-AUC falls from above 0.99 on CIFAKE to about 0.6, not far above random guessing (0.5).
- **ResNet18 calls almost everything "fake"**, including 92% of real photos. Its overall COCOAI accuracy of 81.3% is **misleading**: the test set has 5 fakes for every real photo, so labelling everything "fake" would already score 83%. This is why ROC-AUC and per-class rates are reported instead.
- **Below 0.5 for DALL·E 3 and Midjourney 6:** AUC below 0.5 means the ranking is *inverted*. Both models think these fakes look *more* real than the real photos.
- **SD 2.1 is detected best** (AUC 0.75 to 0.80). It is the closest relative of CIFAKE's Stable Diffusion 1.4.
- **Hypotheses for the collapse:**
  1. **Generator mismatch:** the models learned artefacts specific to SD 1.4. Other families (DALL·E 3, Midjourney) leave different traces.
  2. **Domain mismatch:** "real" in training meant CIFAR-10 (tiny, blurry, old photos). A sharp COCO photo shrunk to 64x64 does not look like CIFAR-10, so ResNet18 treats it as "not real".
  3. **Resolution loss:** COCOAI images are 270 to 1024 pixels wide and are shrunk to 32 or 64 pixels. As section 3 showed, resizing destroys the traces these models rely on.

**Shortcut we removed:** every COCOAI fake is square, while only 3% of the real photos are [`cocoai_subset.json`]. A model given raw image sizes could cheat. Our resize plus centre-crop gives every image the same square shape, so this shortcut is not available. JPEG quality is also the same (about 75) for all classes [`jpeg_quality.json`], so compression level is not a shortcut either.

## 5. Fine-tuning on CIFAKE + COCOAI

**Setup:** each model was fine-tuned for 10 epochs (lr 1e-4), starting from its CIFAKE checkpoint, on 1,600 random CIFAKE training images plus the 1,600-image COCOAI fine-tuning split. That split contains 800 real images and 160 from each generator, with captions disjoint from the test split. Results are on the same held-out COCOAI test split.

| | SmallCNN before → after | ResNet18 before → after |
|---|---|---|
| COCOAI ROC-AUC | 0.581 → 0.742 | 0.610 → **0.878** |
| Real photos called real | 72.9% → 72.3% | 7.7% → 79.5% |
| Real vs DALL·E 3, AUC | 0.433 → 0.731 | 0.520 → 0.946 |
| Real vs Midjourney 6, AUC | 0.429 → 0.579 | 0.437 → 0.830 |
| CIFAKE test accuracy | 96.5% → 94.7% | 97.8% → 96.5% |
| CIFAKE test ROC-AUC | 0.9943 → 0.9883 | 0.9977 → 0.9952 |

[`metrics_*_mix.json`, `cocoai_*_mix.json`]

- **Large gain, small cost:** only 160 examples per generator raised ResNet18's COCOAI AUC from 0.61 to 0.88, at a cost of about 1.3 accuracy points on CIFAKE. Mixing CIFAKE back in kept that "forgetting" small.
- **Capacity matters:** ResNet18 gained far more than SmallCNN.
  - **Hypothesis:** SmallCNN has fewer parameters and sees 32x32 input, which leaves less capacity and less detail for learning several generators at once.
- **Still improving:** both validation AUCs were still rising at epoch 10 (0.892 and 0.947 [`*_mix_train_log.csv`]), so longer fine-tuning might help further. We did not run it.
- **Important caveat:** after fine-tuning, these five generators are **no longer unseen**. The 0.878 figure measures generalisation to *new images* from *known* generators, not to new generators. A truly unseen generator (one released after this dataset) would probably score lower. This was not tested.
- **Not measured:** robustness (section 3) was measured only for the CIFAKE-only models.

## 6. Grad-CAM observations

[`figures/gradcam_examples.png`]

- **ResNet18:** maps are smooth blobs, usually centred on the main object. This is because its last conv layer has a 4x4 map.
- **SmallCNN:** maps for confident FAKE predictions highlight textured or background regions. For confident REAL predictions the map is often empty (all blue): after the ReLU, no region adds positive evidence. This is a known limitation of Grad-CAM.
- **What this does and does not show:** Grad-CAM tells us *where* the evidence is, not *what* it is. It does not show that the models look at "AI artefacts" rather than content.

## 7. What this means for the app

- **Default model:** the app defaults to **ResNet18 + COCOAI fine-tune**, the best measured generalisation. Every result carries warnings about downscaling, compression and the limited training generators.
- **Verdict band:** the wide "uncertain" band (0.35 to 0.65) and the disclaimer are justified by sections 3 and 4. On real-world images these models are far less accurate than their CIFAKE numbers suggest.
