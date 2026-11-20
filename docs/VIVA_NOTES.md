# Viva notes: AI-Generated Image Detector

These notes explain the project in plain language. Every number quoted here comes from a file in `reports/`, and the file is named next to the number.

---

## 1. The project in one paragraph

I trained two image classifiers to answer one question: *"Is this image a real photo or AI-generated?"* The first is a small CNN trained from scratch on CPU. The second is an ImageNet-pretrained ResNet18 that I fine-tuned. Both were trained on **CIFAKE**: 60,000 real CIFAR-10 photos and 60,000 Stable Diffusion 1.4 fakes, all 32x32 pixels. I then checked how trustworthy they are in four ways:

1. **Calibration:** are the probabilities honest?
2. **Robustness:** what happens after JPEG compression, resizing or blur?
3. **Generalisation:** what happens on images from five newer generators in the COCOAI dataset, which the models have never seen?
4. **Explanation:** Grad-CAM heatmaps showing where each model looked.

A FastAPI web app on localhost lets a user upload an image and see a calibrated probability, a three-band verdict, the heatmap and honest warnings.

---

## 2. Module by module

### `src/data.py`: getting images into the model
- **Listing files:** files are listed and **sorted** so the random split is reproducible. The file system does not guarantee any order.
- **Split:** the official 100k training set is split 90% train and 10% validation with `train_test_split(stratify=labels, random_state=42)`. *Stratified* means both parts keep the 50/50 real/fake ratio. The official 20k **test set is never used** for training, early stopping or calibration. It is used only once, for the final numbers.
- **Saved split:** the split is saved to `data/splits.json`, so every later script uses exactly the same validation images.
- **Transforms:** every image goes through *resize the shorter side, then centre-crop* to the model's size (32 or 64), then `ToTensor` (pixels to floats in [0, 1]), then `Normalize` (subtract the mean, divide by the std).
- **Augmentation:** on training data **only**, a random horizontal flip and a light random crop. It is light because heavy augmentation could erase the tiny pixel artefacts that reveal AI generation.
- **Mixing datasets:** `ImageDataset` reads images lazily, one batch at a time. `make_loaders` can also build a **mix** of CIFAKE and COCOAI for fine-tuning.

### `src/models.py`: the two networks
- **SmallCNN:** 3 blocks of *conv 3x3, BatchNorm, ReLU, MaxPool 2x2*, then *Flatten, Linear(2048 to 128), ReLU, Dropout(0.3), Linear(128 to 1)*. Shapes go 3x32x32, 32x16x16, 64x8x8, 128x4x4. It has 355,873 parameters (`reports/train_smallcnn.json`).
- **ResNet18Detector:** torchvision ResNet18 with ImageNet weights. The final 1000-class layer is replaced by `Linear(512 to 1)`, and the first max-pool is removed so a 64x64 input still leaves a 4x4 map at the end, which Grad-CAM needs. It has 11,177,025 parameters.
- **One output:** both output **one logit** *z*, and *P(fake) = sigmoid(z)*.
- **Safe loading:** checkpoints are loaded with `torch.load(weights_only=True)`, so a tampered file cannot run code.

### `src/train.py`: the training loop
For each epoch:
1. For each batch: forward pass, compute the loss, `loss.backward()` (backpropagation), then `optimizer.step()` (Adam).
2. Evaluate loss, accuracy and ROC-AUC on validation.
3. If validation ROC-AUC improved, save the checkpoint. If it has not improved for `patience` epochs, stop. This is **early stopping**: it keeps the best model and avoids overfitting.

Seeds are fixed (Python, NumPy, PyTorch), so a run can be repeated. Epoch 1 of SmallCNN gave exactly the same validation AUC (0.9753) in two separate runs. Every epoch is logged to CSV and plotted.

### `src/evaluate.py`: final numbers
- **Test metrics:** accuracy, precision, recall, F1, ROC-AUC, the confusion matrix, the ROC curve, and CPU latency per image (batch size 1, mean of 200 runs).
- **Cross-generator mode (`--cocoai`):** evaluates on the held-out COCOAI split, both overall and *real vs each generator*. Each per-generator comparison is balanced: 1000 vs 1000.

### `src/calibrate.py`: honest probabilities
Fits one number, the temperature *T*, on the **validation** set, then reports ECE and NLL before and after on the **test** set. It also draws a reliability diagram.

### `src/robustness.py`: real-world damage
Re-runs the test set after:
- JPEG re-compression at quality 90, 70, 50 and 30
- shrinking to 75% or 50% and enlarging back
- Gaussian blur with radius 0.5 or 1.0

This simulates what WhatsApp or Instagram do to a shared image.

### `src/gradcam.py`: where did the model look?
Hand-written Grad-CAM using a forward hook, plus a tensor hook that captures the gradient. See the maths in section 3.

### `src/infer.py`: one image, safely
1. **Format and pixel count first:** Pillow reads only the header, and the file is refused if it is not JPEG/PNG/WebP or has more than 25 MP. This stops *decompression bombs*: small files that expand to gigabytes when decoded.
2. **Integrity:** `verify()` checks the file structure, which catches corrupt files.
3. **Orientation:** `exif_transpose` makes phone photos upright.
4. **Clean copy:** the image is rebuilt from its pixel array, which drops all EXIF metadata (GPS, camera serial numbers).
5. **Prediction:** preprocess, run Grad-CAM and get the logit, then compute *P(fake) = sigmoid(z / T)*.
6. **Verdict band:** below 0.35 is *likely real*, 0.35 to 0.65 is *uncertain*, above 0.65 is *likely AI-generated*.
7. **Warnings:** downscaling, centre-crop, heavy JPEG compression (quality estimated from the file's quantisation table), and limited training generators.

### `app/main.py` and `app/static/index.html`: the web app
- **Binding:** FastAPI is bound to **127.0.0.1**, so it is only reachable from this computer.
- **`/predict` checks:** the declared type is checked first (415 if wrong). It reads at most 5 MB + 1 byte (413 if larger). Pillow then checks the real format and integrity (422 if bad). Uploads are never written to disk.
- **`/models`:** reads each model's numbers from `reports/*.json`, so the UI never shows a hardcoded number.
- **UI:** one HTML file with vanilla JS. It has drag-and-drop upload, a model selector, a result card, a heatmap opacity slider, and the disclaimer.

### `scripts/`
- `inspect_data.py`: class balance, split sizes, leakage check, sample grid.
- `prepare_cocoai.py`: builds the balanced COCOAI subset and **splits by caption**, so the same scene never appears in both training and test.
- `jpeg_quality_check.py`: checks that real and fake images do not differ in JPEG quality. If they did, a model could learn compression level instead of AI artefacts.
- `kaggle_train.py`: trains ResNet18 on Kaggle's free GPU.
- `report_tables.py`: builds every results table from the JSON files.

---

## 3. The maths

### 3.1 Sigmoid and binary cross-entropy (the loss)
The model outputs a logit *z* (any real number), and

&nbsp;&nbsp;&nbsp;&nbsp;*p = sigmoid(z) = 1 / (1 + e^(-z))*, which lies in (0, 1).

For a label *y* (1 = fake, 0 = real), the **binary cross-entropy** is

&nbsp;&nbsp;&nbsp;&nbsp;*L = -[ y log p + (1 - y) log(1 - p) ]*

- **Fake image (y = 1):** *L = -log p*. It is 0 when p = 1 and grows without limit as p goes to 0.
- **Real image (y = 0):** *L = -log(1 - p)*.
- **Why log:** confident wrong answers are punished much more than unsure ones.
- **Statistical view:** minimising the average *L* is the same as maximising the likelihood of the training labels.
- **In code:** `BCEWithLogitsLoss` combines the sigmoid and the log in one numerically stable formula.
- **Useful fact:** dL/dz = p - y. The gradient is simply "prediction minus truth".

### 3.2 Backpropagation
The network is a chain of functions, *z = f3(f2(f1(x)))*. The chain rule gives the gradient of the loss for every weight:

&nbsp;&nbsp;&nbsp;&nbsp;*dL/dW1 = dL/dz · dz/df2 · df2/df1 · df1/dW1*

- **How it runs:** backprop computes these products from the output backwards, reusing each intermediate result, so one backward pass costs about as much as one forward pass. PyTorch records the operations during the forward pass (the autograd graph), and `loss.backward()` walks the graph in reverse.
- **Adam update:** Adam then updates each weight with *W := W - lr · m̂ / (√v̂ + ε)*.
  - *m̂* is a running average of the gradients (momentum).
  - *v̂* is a running average of the squared gradients, which gives each weight its own step size.

### 3.3 Batch normalisation
For each channel, over a mini-batch:

&nbsp;&nbsp;&nbsp;&nbsp;*x̂ = (x - μ_batch) / √(σ²_batch + ε)*, then *y = γ x̂ + β*

- **What it does:** it keeps every layer's inputs at a stable scale (mean 0, variance 1). That allows higher learning rates and faster, more stable training.
- **Learned parameters:** γ and β are learned, so the network can undo the normalisation if that helps.
- **Training vs testing:** during training the batch statistics are used. At test time (`model.eval()`) running averages collected during training are used, so one image's prediction does not depend on the other images in its batch.
- **No conv bias:** a conv bias is redundant before BatchNorm, because β does the same job. That is why the convs use `bias=False`.

### 3.4 ROC curve and ROC-AUC
- **Threshold sweep:** sweep the decision threshold *t* from 1 down to 0, and at each *t* plot:
  - TPR = TP / (TP + FN), the fraction of fakes caught (recall), against
  - FPR = FP / (FP + TN), the fraction of real images wrongly flagged.
- **AUC:** the area under that curve.
- **Meaning:** AUC is the **probability that a randomly chosen fake gets a higher score than a randomly chosen real image**. 0.5 is random guessing and 1.0 is perfect ranking.
- **Why it is useful:** it does not depend on the threshold, and it is not distorted by class imbalance. That is why early stopping uses it, and why it is the main metric on COCOAI, where the overall set has 5 fakes for every real image.

### 3.5 Calibration, ECE and temperature scaling
- **Calibrated means:** among all images where the model says "90% fake", about 90% really are fake.
- **ECE (Expected Calibration Error):** group predictions into *B* = 15 bins by confidence (the probability of the predicted class, between 0.5 and 1). Then

  &nbsp;&nbsp;&nbsp;&nbsp;*ECE = Σ_b (n_b / N) · | accuracy_b - confidence_b |*

  The **reliability diagram** plots accuracy_b against confidence_b. A perfectly calibrated model sits on the diagonal.
- **Temperature scaling (Guo et al., 2017):** use *p = sigmoid(z / T)*, with one parameter *T > 0* fitted on the **validation** set by minimising NLL (a grid search over 2,000 values from 0.05 to 20).
  - T > 1 softens over-confident probabilities. T < 1 sharpens under-confident ones.
  - Dividing by a positive number never changes the sign of *z*, so **accuracy, the confusion matrix and ROC-AUC do not change**. Only the confidence changes.
- **Why validation and not test:** if T were fitted on test, the "after" ECE would be optimistically biased. That would be a form of data leakage.

### 3.6 Grad-CAM
Let *A^k* be the *k*-th feature map of the last conv layer (8x8 in SmallCNN, 4x4 in ResNet18), and let *s* be the score being explained: *s = z* for a FAKE verdict and *s = -z* for a REAL verdict.

1. **Channel weights:** α_k = (1/HW) Σ_i,j ∂s/∂A^k_ij. Each is the average gradient, meaning how much channel *k* pushes the score up.
2. **Map:** *L = ReLU( Σ_k α_k A^k )*. The ReLU keeps only the evidence *for* the decision.
3. **Display:** upsample *L* bilinearly to the input size, scale it to [0, 1], colour it (blue = low, red = high) and overlay it on the image.

In code, a **forward hook** saves *A* and registers a **tensor hook** that saves ∂s/∂A during `backward()`.

**Limitation:** Grad-CAM shows *where* the evidence was, not *what* the evidence is. With 4x4 or 8x8 maps it is coarse.

---

## 4. Key results to remember

All of these come from `reports/` (see `reports/results.md` for the full tables).

| | SmallCNN | ResNet18 | ResNet18 + COCOAI fine-tune |
|---|---|---|---|
| CIFAKE test accuracy | 96.5% | 97.8% | 96.5% |
| CIFAKE test ROC-AUC | 0.994 | 0.998 | 0.995 |
| COCOAI ROC-AUC | 0.581 | 0.610 | 0.878 |
| COCOAI real photos called real | 72.9% | 7.7% | 79.5% |
| Accuracy after 75% down-up resize | 64.9% | 64.5% | not measured |
| Parameters | 355,873 | 11,177,025 | 11,177,025 |
| CPU time per image | 0.23 ms | 3.14 ms | n/a |

**One-sentence summary:** *excellent on the dataset the models were trained on, much weaker on new generators and on resized or blurred images. A little data from the new generators fixes much of the generator gap.*

---

## 5. Likely viva questions, with short answers

**Q1. Why did you choose CIFAKE?**
It is free (MIT licence), balanced (60k real and 60k fake), and small (32x32), so a model can be trained on a laptop CPU. SmallCNN took about 40 s per epoch. The weakness is that its fakes all come from one generator, Stable Diffusion 1.4, which is why I also tested on COCOAI.

**Q2. How did you split the data, and why does the test set stay untouched?**
I split the official training set 90% train and 10% validation, stratified, with seed 42. The validation set is used for early stopping and for fitting the temperature. The official 20,000-image test set is used only for final numbers. If I tuned anything on test data, the test score would be optimistic, because I would have indirectly fitted to it.

**Q3. Your SmallCNN gets 96.5%. Is that good?**
On CIFAKE, yes. ROC-AUC is 0.994, and errors are balanced (390 false positives, 311 false negatives). But the same model scores only 0.581 AUC on COCOAI, so the 96.5% does **not** mean "it detects AI images in general".

**Q4. Why use ROC-AUC as well as accuracy?**
Accuracy depends on the 0.5 threshold and on the class mix. On COCOAI the test set has 5 fakes for every real image, so ResNet18 scored 81.3% accuracy while calling 92% of real photos fake; labelling everything "fake" would score 83%. ROC-AUC measures how well the model *ranks* fakes above reals. It ignores the threshold and the imbalance, and was 0.610 there, which tells the true story.

**Q5. What is early stopping and did it trigger?**
Training stops when validation ROC-AUC has not improved for `patience` epochs, and the best checkpoint is kept. ResNet18 stopped after 6 of 8 epochs, with the best at epoch 4 and patience 2. SmallCNN's best epoch was 12; it ran all 15 epochs, but epochs 13 to 15 did not improve on epoch 12.

**Q6. Why fine-tune a pretrained ResNet18 instead of training it from scratch?**
ImageNet weights already detect edges and textures, so training converges fast. Validation AUC was 0.994 after one epoch. 11 million parameters trained from scratch on 90k tiny images would overfit more easily. I removed the first max-pool so a 64x64 input still has a 4x4 map at the end. I used a smaller learning rate (3e-4) so the pretrained features are adjusted gently rather than destroyed.

**Q7. Was ResNet18 worth it?**
On CIFAKE it gained only 1.3 accuracy points for 31x more parameters and about 14x slower CPU inference (3.14 ms vs 0.23 ms). The real benefit appeared when fine-tuning on COCOAI: ResNet18 reached 0.878 AUC, against SmallCNN's 0.742.

**Q8. What is temperature scaling and what did it do here?**
It divides the logit by one learned number T before the sigmoid, with T fitted on validation data by minimising NLL. It changes confidence, not decisions, so accuracy and AUC stay the same.
- **Original models:** T was close to 1 (0.952 and 1.029) and ECE barely moved (for example 0.0049 to 0.0041), because they were already well calibrated.
- **Fine-tuned models:** it helped clearly. SmallCNN_mix went from ECE 0.0373 to 0.0086 (T = 0.62, sharpening under-confident outputs). ResNet18_mix went from 0.0147 to 0.0033 (T = 1.50, softening over-confident outputs).

**Q9. What is ECE?**
Group predictions by confidence into 15 bins. In each bin, compare the average confidence with the actual accuracy. ECE is the size-weighted average gap. An ECE of 0.004 means the stated confidence is off by about 0.4 percentage points on average.

**Q10. What happened in the robustness test?**
- **JPEG:** mostly harmless down to quality 50. ResNet18 went from 97.8% to 96.0%.
- **Resize and blur:** devastating. A 75% down-up resize gave 64.5%, and a radius-1.0 blur gave 59.3%.
- **Kind of error:** almost all new errors were fakes called real. ResNet18's recall fell from 98.2% to 29.4% after the 75% resize.
- **My hypothesis:** the models rely on fine, high-frequency traces of the generator, and smoothing removes them. I have not tested this.

**Q11. Why did both models fail on COCOAI, and why is ResNet18 so biased toward "fake"?**
These are hypotheses, not proven:
1. The models learned SD 1.4-specific traces, and other generators leave different ones.
2. "Real" in training meant CIFAR-10 (tiny, blurry photos), so a sharp COCO photo shrunk to 64x64 looks unfamiliar and gets called fake.
3. Shrinking 270 to 1024 px images to 32 or 64 px destroys the traces.

SD 2.1, the closest relative of SD 1.4, was detected best (AUC 0.75 to 0.80). DALL·E 3 and Midjourney fell **below** 0.5, meaning the ranking was inverted.

**Q12. How did you stop the COCOAI test from being "contaminated"?**
Each COCO caption has one real photo and five AI images of the same scene. I split by **caption** using a hash, so no scene appears in both the fine-tuning and test splits. I also checked for shortcuts:
- All COCOAI fakes are square, while only 3% of real photos are. The centre-crop removes that cue.
- JPEG quality is about 75 for every class in both datasets.

**Q13. Does the 0.878 after fine-tuning prove the model generalises?**
No. After fine-tuning, those five generators are no longer unseen. 0.878 measures new images from *known* generators. A generator released later would probably score lower, and I did not test that. Fine-tuning used only 160 images per generator, plus 1,600 CIFAKE images to limit forgetting. CIFAKE accuracy dropped only from 97.8% to 96.5%.

**Q14. How does Grad-CAM work, and what does your heatmap show?**
It takes the gradient of the score with respect to the last conv layer's maps, averages each channel's gradient into a weight, sums the weighted maps, and applies a ReLU. I implemented it with a forward hook and a tensor gradient hook. For a FAKE verdict I explain *z*, and for a REAL verdict I explain *−z*. It shows *where* the evidence is, not *what* it is. The maps are coarse: 8x8 for SmallCNN and 4x4 for ResNet18. For confident REAL predictions SmallCNN's map is often empty, which is a known Grad-CAM limitation.

**Q15. How is the app made safe, and how is it made ethical?**
**Safety:**
- It binds to 127.0.0.1, so other machines cannot reach it.
- It reads at most 5 MB, and the type and real format are checked: JPEG, PNG or WebP only.
- The pixel count is checked before decoding (25 MP cap against decompression bombs), and `verify()` catches corrupt files.
- EXIF metadata is dropped by rebuilding the image from its pixels, and nothing is written to disk.
- Checkpoints load with `weights_only=True`.
- Downloaded datasets are only read by scripts run with `python -I`.

**Ethics:**
- A three-band verdict with a wide "uncertain" zone (0.35 to 0.65).
- Calibrated probabilities.
- Per-image warnings: downscaling, centre crop, heavy compression, limited generators.
- The headline numbers in the UI are read from the metric files.
- The disclaimer: "This is a probability estimate, not proof. Do not use it to accuse anyone."

**Bonus: what would you do with more time?**
- Fine-tune longer (both models' validation AUC was still rising at epoch 10).
- Train on larger inputs so less detail is destroyed.
- Add resize, blur and JPEG augmentation during training to fix the robustness drop.
- Test on a generator that was held out completely.
- Measure calibration on COCOAI, not only on CIFAKE.
