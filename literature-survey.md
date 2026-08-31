# Literature Review: Deep Learning for Pediatric Pneumonia CXR Classification (Kermany)

## Abstract

This survey covers 10 papers on automated pneumonia classification from pediatric chest radiographs, centered on the Guangzhou Women and Children’s Medical Center collection released by Kermany et al. (2018) and redistributed on Kaggle as Chest X-Ray Images (Pneumonia). Only papers we could read in full (local PDF, arXiv, or open PMC/MDPI) are included. Extra ImageNet ensembles that repeat the same binary result were left out. The corpus is small (~5,856 frontal CXRs, ages 1–5), three-way labeled (normal / bacterial pneumonia / viral pneumonia), and badly imbalanced: pneumonia outnumbers normal by roughly 3:1, and bacterial cases outnumber viral.

The dominant recipe is the same pattern the APTOS diabetic-retinopathy literature settled on: an ImageNet-pretrained CNN (Inception, VGG, ResNet, DenseNet, later EfficientNet) is transferred onto Kermany; accuracy, sensitivity/recall, specificity, F1, and AUC are reported; the 624-image patient-independent test split from the Cell paper is used by some later authors and quietly replaced by others. Binary normal-versus-pneumonia accuracy commonly lands between 93% and 99%. Three-class and bacterial-versus-viral accuracy drop several points. Viral pneumonia is the hardest class, analogous to Severe/Proliferative grades on APTOS.

The source paper itself is now in this folder (`kermany_research_paper.pdf`). Direct extraction changes two facts that later comparison tables blur. First, Kermany et al. **froze** Inception V3 convolutional layers and retrained only the softmax; they report that unfreezing (“fine-tuning”) *decreased* accuracy through overfitting. Second, pneumonia-versus-normal **specificity is 90.1%**, not precision — Rahman et al. 2020 Table 5 mislabels that number.

Two caveats cut across the field. First, the Cell paper uses a 5,232 / 624 split; Kaggle later peeled 16 training images into a toy validation folder, and many “99%” papers reshuffle train entirely. Those numbers are not comparable to papers that keep the 624-patient test. Second, almost every study is single-center pediatric data; external tests (RSNA, CheXpert, other hospitals) show a sharp drop. The useful research questions for a semester project are therefore not “can a CNN beat 90% on Kermany?” — that is settled — but how to handle the split, the viral class, leakage, and honest metrics.

## Key Papers

For each paper: backbone, imbalance handling, preprocessing, and metrics. Kermany is the CXR analog of APTOS; the headline metric is **accuracy plus sensitivity/AUC**, not quadratic weighted kappa.

### 1. Kermany et al., 2018 — *Cell*  *(local PDF: `kermany_research_paper.pdf`)*

**Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.**  
Daniel S. Kermany, Michael Goldbaum, Wenjia Cai, Carolina C. S. Valentim, Huiying Liang, Sally L. Baxter, et al. (Kang Zhang, lead contact). *Cell* 172(5):1122–1131.e2.  
DOI: [10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010) · PMID: 29474911  
**Type:** Application / methods. Primary task is retinal OCT; pediatric CXR is the generalization experiment that created this project’s dataset.  
**Access:** full PDF in the project folder (20 pages, including STAR Methods and Figure S6). Numbers below are from that file, not from later citation tables.

#### Overview

Clinical-decision support for imaging is hard to train and hard to explain. Kermany et al. show that an ImageNet-pretrained Inception V3 can be adapted with transfer learning — frozen convolutional “bottlenecks,” a new softmax — and reach expert-level referral on OCT (CNV, DME, drusen, normal). They then apply the **same** framework to pediatric chest X-rays to argue the method is not OCT-specific.

On CXR the clinical claim is triage, not just a label. Bacterial pneumonia is an **urgent referral** (antibiotics); viral pneumonia is **supportive care**; normal is **observation**. That mapping is why later 3-class work matters more than binary accuracy.

#### Backbone, imbalance, preprocessing, metrics

- **Backbone:** Inception V3 pretrained on ImageNet (Szegedy et al., 2016), implemented in TensorFlow. Convolutional layers **frozen** as fixed feature extractors; bottleneck activations cached; **only the final softmax is trained from scratch**. Adam, learning rate 0.001, batches of 1,000 images/step, 10,000 steps (100 epochs) on Ubuntu 16.04 + GTX 1080 8 GB. Holdout after every step; best model kept. **Unfreezing the conv layers and fine-tuning with backpropagation decreased performance (overfitting).** Figure 1.
- **Imbalance:** No class weights, SMOTE, or geometric augmentation are described for CXR. They rely on transfer learning plus a 3:1 pneumonia:normal train set (3,883 vs 1,349) and 2:1 bacterial:viral (2,538 vs 1,345).
- **Preprocessing / labels:** Anterior-posterior films, children aged **1–5**, Guangzhou Women and Children’s Medical Center, routine care. Low-quality / unreadable scans removed. Two expert physicians graded diagnoses; a third expert checked the evaluation set. IRB / HIPAA / Helsinki. No lung crop or CLAHE.
- **Split (from the paper, not Kaggle):** **5,232** train images (1,349 normal; 2,538 bacterial; 1,345 viral) from 5,856 patients. Test: **234 normal + 390 pneumonia (242 bacterial, 148 viral) from 624 patients**. Patient-independent holdout. The later Kaggle layout `5,216 / 16 / 624` peels 16 images out of that train set — the 16-image “val” folder is **not** in the Cell protocol.
- **Metrics (CXR, paper text + Figure 6):**
  - Pneumonia vs normal: accuracy **92.8%**, sensitivity **93.2%**, specificity **90.1%**, AUC **96.8%**.
  - Bacterial vs viral: accuracy **90.7%**, sensitivity **88.6%**, specificity **90.9%**, AUC **94.0%**.
- **Why it matters:** This *is* the public pediatric pneumonia CXR set (Mendeley DOI [10.17632/rscbjbr9sj.3](https://doi.org/10.17632/rscbjbr9sj.3)). Every later paper trains on this collection the way APTOS papers train on APTOS 2019.

#### Highlights

- Same transfer recipe works on OCT and on a “difficult” CXR task (lots of non-lung clutter).
- Bacterial vs viral is already ~2 points worse than binary pneumonia detection — the hard-class pattern starts here.
- Figure S6 states the radiology the model is supposed to use: bacterial = **focal lobar consolidation**; viral = **diffuse interstitial pattern**; normal = clear lungs.
- Occlusion testing (20×20 kernel) is reported for OCT (94.7% ROI hit rate), not for CXR.
- Code and images released; no competing interests declared.

#### Strengths

- Patient-independent test set sized for 0.05 marginal error / 95% CI (sample-size note is for OCT but the CXR test is also patient-held-out).
- Clinical referral labels, not just class names.
- Honest about transfer-learning limits: a randomly initialized net on “unlimited” OCT would eventually beat this, but would take weeks; their multi-class holdout finished in under 2 hours.

#### Weaknesses

- CXR is a generalization paragraph plus Figure 6, not a full CXR methods paper. No CXR confusion matrix, no 3-class joint accuracy, no class-wise viral recall, no CXR occlusion maps.
- Single hospital, ages 1–5 only.
- Frozen-backbone choice is the 2018 answer; the 2026 comparative study on this same set finds fine-tuning last blocks **+5.5%**. That disagreement is now a real ablation, not a rumor.
- Rahman et al. 2020 Table 5 quotes this paper as precision 90.1%; the Cell text says **specificity** 90.1%. Use the PDF.

#### Reproducibility

High for data (public JPEGs). Medium for the exact CXR trainer: STAR Methods specify Inception V3, frozen conv, Adam 0.001, 100 epochs, but not CXR input size, augmentation, or the exact train/val curves’ image counts beyond Figure 6.

---

### 2. Rajaraman et al., 2018 — *Applied Sciences*

**Visualization and Interpretation of Convolutional Neural Network Predictions in Detecting Pneumonia in Pediatric Chest Radiographs.**  
Sivaramakrishnan Rajaraman, Sema Candemir, Incheol Kim, George Thoma, Sameer Antani. *Appl. Sci.* 8(10):1715.  
DOI: [10.3390/app8101715](https://doi.org/10.3390/app8101715) · PMC: [PMC7250407](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7250407/)

- **Backbone:** Customized VGG16 (best). Also sequential, residual, and Inception CNNs.
- **Imbalance:** No explicit reweighting. Noisy images kept in training “to reduce bias and overfitting.”
- **Preprocessing:** Quality screen; lung ROI from an anatomical-atlas detector; whole CXR and cropped boxes resampled to **1024×1024** and mean-normalized. No geometric augmentation reported.
- **Metrics (custom VGG16, cropped ROI):** pneumonia vs normal — acc **96.2%**, AUC **0.993**, recall **96.2%**, F1 **0.970**. Bacterial vs viral — acc **93.6%**, AUC **0.962**, recall **98.4%**, specificity **86.0%**.
- **Why it matters:** First strong explainability paper on this set (average-CAM, LIME). Shows that **cropping to lungs** beats using the full film.

**Highlights:** Localization, not just a score. Bacterial/viral specificity (0.86) already reveals the hard class.  
**Weaknesses:** Atlas lung detector under-segments costophrenic angles. Residual/Inception over-parameterized for this small, low-variance pediatric set.

---

### 3. Stephen et al., 2019 — *Journal of Healthcare Engineering*

**An Efficient Deep Learning Approach to Pneumonia Classification in Healthcare.**  
Okeke Stephen, Mangal Sain, Uchenna Joseph Maduh, Do-Un Jeong. *J. Healthc. Eng.* 2019:4180949.  
DOI: [10.1155/2019/4180949](https://doi.org/10.1155/2019/4180949) · PMC: [PMC6458916](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6458916/)

- **Backbone:** Custom CNN trained **from scratch** (conv + max-pool + ReLU + flatten + dropout + dense + sigmoid). No ImageNet.
- **Imbalance:** Aggressive Keras augmentation instead of class weights.
- **Preprocessing / aug:** rescale 1/255; rotation 40°; width/height shift 0.2; shear 0.2; zoom 0.2; horizontal flip. Best reported size **200×200×3**.
- **Metrics:** training acc **95.31%**, validation acc **93.73%**. Binary only. They reshuffled into 3,722 train / 2,134 val — **not** the official 624-image test.
- **Why it matters:** Proof that a small scratch CNN plus augmentation is enough for binary Kermany. Baseline for “do we need transfer learning?”

**Highlights:** Simple, reproducible, no pretrained weights.  
**Weaknesses:** Binary only. Custom split. No AUC/sensitivity table. Limited depth of data, as the authors note.

---

### 4. Rahman et al., 2020 — arXiv / *Applied Sciences*

**Transfer Learning with Deep Convolutional Neural Network (CNN) for Pneumonia Detection using Chest X-ray.**  
Tawsifur Rahman, Muhammad E. H. Chowdhury, et al.  
arXiv: [2004.06578](https://arxiv.org/abs/2004.06578) (full text read)

- **Backbone:** AlexNet, ResNet18, DenseNet201, SqueezeNet. **DenseNet201 wins all three tasks.**
- **Imbalance:** Geometric augmentation (rotate 45°, 10% scale, 10% translate) until each class has **4,500** train images.
- **Preprocessing:** Resize 227×227 (AlexNet/SqueezeNet) or 224×224 (ResNet/DenseNet); pretrained-model normalization.
- **Metrics (DenseNet201):**
  - Normal vs pneumonia: acc **98.0%**, recall **99%**, spec **97%**, AUC **0.98**, F1 **0.981**
  - 3-class: acc **93.3%**, recall **93.2%**, AUC **0.95**
  - Bacterial vs viral: acc **95.0%**, recall **96%**, AUC **0.952**
- **Why it matters:** The only early paper that reports **all three task formulations** side by side. This is the cleanest analog to “APTOS 5-grade vs binary referable DR.”

**Highlights:** Confusion matrices show bacterial ↔ viral as the main error. Activation-map inspection.  
**Weaknesses:** MATLAB 2019a, 5-fold on an augmented pool; test sets are small (~200/class). Binary numbers look better than 3-class for a reason.

---

### 5. Salehi et al., 2021 — *British Journal of Radiology*

**Automated detection of pneumonia cases using deep transfer learning with paediatric chest X-ray images.**  
Mohammad Salehi, Reza Mohammadi, Hamed Ghaffari, Nahid Sadighi, Reza Reiazi. *Br. J. Radiol.* 94:20201263.  
DOI: [10.1259/bjr.20201263](https://doi.org/10.1259/bjr.20201263) · PMC: [PMC8506182](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8506182/)

- **Backbone:** VGG19, DenseNet121, Xception, ResNet50. Best: **DenseNet121**.
- **Imbalance:** **SMOTE** (rare in this literature; most papers only augment).
- **Preprocessing:** rescale 1/255, rotation 15°, small shifts/shear/zoom, flips, noise reduction, contrast enhancement, per-channel mean subtraction.
- **Metrics:** all models ≥83% acc. DenseNet121 **86.8%** acc, AUC **0.86**; Xception 86.0% / 0.81. Sensitivity **>91%** for all.
- **Why it matters:** A radiology-journal paper with **honest, lower** numbers. Useful counterweight to 99% Kaggle papers.

**Highlights:** Authors warn that same-patient images may leak across splits, so published accuracy is “likely an overestimate.” No laterals (needed in ~15% of real reads).  
**Weaknesses:** ResNet50 overfit. Binary only. Still single-center.

---

### 6. Ayan, Karabulut & Ünver, 2021 — *Arabian Journal for Science and Engineering*

**Diagnosis of Pediatric Pneumonia with Ensemble of Deep Convolutional Neural Networks in Chest X-Ray Images.**  
Enes Ayan, Bergen Karabulut, Halil Murat Ünver. *Arab. J. Sci. Eng.* 47:2123.  
DOI: [10.1007/s13369-021-06127-z](https://doi.org/10.1007/s13369-021-06127-z) · PMC: [PMC8435166](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8435166/)

- **Backbone:** Ensemble of VGG-16/19, ResNet-50, Inception-V3, Xception, MobileNet, SqueezeNet, plus a custom **PNet**. Best ensemble used **three** models, not seven.
- **Imbalance:** 1,349 normal images augmented (rotate/zoom/flip) to **2,534** to approach the bacterial count.
- **Preprocessing:** Transfer learning, global average pooling, batch norm, L2.
- **Metrics:** Binary ensemble — acc **95.83%**, sens **97.76%**, spec **92.73%**, AUC **95.21%**. **3-class ensemble acc 90.71%**. Viral pneumonia recall only **77.03%** (F1 82.31%); bacterial recall 97.93%.
- **Why it matters:** Best published picture of the **viral class failure**. This is the Kermany equivalent of “Severe and Proliferative DR are hardest.”

**Highlights:** They measured the effect of class distribution. Bigger ensembles (5–7) were worse than a group of 3.  
**Weaknesses:** Hyperparameters by trial-and-error. Official-looking 624 test, but viral remains unsolved.

---

### 7. Kundu et al., 2021 — *PLOS ONE*

**Pneumonia detection in chest X-ray images using an ensemble of deep learning models.**  
Rohit Kundu, Ritacheta Das, Zong Woo Geem, Gi-Tae Han, Ram Sarkar. *PLoS One* 16(9):e0256630.  
DOI: [10.1371/journal.pone.0256630](https://doi.org/10.1371/journal.pone.0256630) · PMC: [PMC8423280](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8423280/)

- **Backbone:** Weighted-probability ensemble of **GoogLeNet + ResNet-18 + DenseNet-121**.
- **Imbalance:** Transfer learning for small data; no SMOTE.
- **Preprocessing:** Resize **224×224×3**.
- **Metrics:** Kermany — acc **98.81%**, precision **98.82%**, recall **98.80%**, F1 **98.79%**, AUC **98.35%**. Same pipeline on **RSNA**: acc **86.95%**, recall **87.02%**.
- **Why it matters:** The first widely cited paper that **takes the model off Kermany**. The 12-point drop is the generalizability result this field needed.

**Highlights:** Two-dataset evaluation. Weighted average, not majority vote.  
**Weaknesses:** Triple training cost. Fails on low-contrast and early infiltrates. Binary.

---

### 8. XCCNet, 2024 — arXiv

**An Explainable Contrastive-based Dilated Convolutional Network with Transformer for Pediatric Pneumonia Detection.**  
arXiv: [2410.16143](https://arxiv.org/abs/2410.16143) (full text read)

- **Backbone:** Hybrid — dilated spatial CNN (SFx) + contrastive transformer (CoTFx), fused into a 2-layer classifier.
- **Imbalance:** **Adversarial data augmentation (ADA)** — GAN-style synthetic CXRs. On Kermany they synthesize 2,624 extra **normal** images (1,583 → 4,207) to match 4,273 pneumonia.
- **Preprocessing:** Full **CXP** stack — contrast/CLAHE, lung segmentation (ResUNet++), **rib suppression**.
- **Metrics (Kermany):** acc **99.76%**, precision/recall/F1 **99.75%**. Also evaluated on VinDr-PCXR, NIH-Pediatric, Trivedi. Ablation: no aug 96.29%; traditional aug 97.18%; full stack 99.76%.
- **Why it matters:** Moves past “resize and fine-tune.” Preprocessing + synthetic minority + hybrid backbone + XAI (Grad-CAM, Score-CAM, LIME).

**Highlights:** Four-dataset claim. Ablations isolate each module.  
**Weaknesses:** Headline 99.76% is on a balanced, heavily preprocessed, 80/10/10 split — not the official 624 test. No severity grading. Needs labeled lungs/ribs pipeline.

---

### 9. Chauhan, Gupta & Doja, 2025 — arXiv

**LightPneumoNet: Lightweight Pneumonia Classifier.**  
Neilansh Chauhan, Piyush Kumar Gupta, Faraz Doja.  
arXiv: [2510.11232](https://arxiv.org/abs/2510.11232) (full text read)

- **Backbone:** Custom 4-block CNN, **388,082** params, 1.48 MB. Trained from scratch. Filters 16→32→64→128.
- **Imbalance:** **`class_weight`** — Normal 2.0, Pneumonia 1.2.
- **Preprocessing:** Resize 224×224, **RGB→grayscale** (they say this helped), [0,1] norm. Aug: rotate 12°, zoom/shift/shear 0.15. No horizontal flip (anatomical left/right).
- **Metrics (official-style independent test):** acc **94.2%**, precision **0.92**, recall **0.99**, F1 **0.96**.
- **Why it matters:** Contrasts the 20–80 M parameter ensembles. Relevant if the project must run on a lab GPU or a clinic PC.

**Highlights:** High recall at tiny size. Explicit class weights. Gray input.  
**Weaknesses:** Precision 0.92. Binary. Small dataset acknowledged.

---

### 10. 2026 — arXiv

**Pediatric Pneumonia Detection from Chest X-Rays: A Comparative Study of Transfer Learning and Custom CNNs.**  
arXiv: [2601.00837](https://arxiv.org/abs/2601.00837) (full text read)

- **Backbone:** Custom 4-block CNN vs ResNet50 / DenseNet121 / EfficientNet-B0, each **frozen** or **fine-tuned** (last two blocks, differential LR 1e-4 / 1e-3).
- **Imbalance:** They do **not** reweight. They report the 2.87:1 ratio and show ResNet50 still balances F1 (99.61% pneumonia vs 98.89% normal).
- **Preprocessing:** 224×224, grayscale→3-channel, ImageNet mean/std. Train aug: flip, ±10° rotate, affine, color jitter.
- **Split:** They discard the official 16-image val set and make a stratified **80/10/10** from the 5,216 train images (4,172 / 521 / 523).
- **Metrics (ResNet50 fine-tune):** acc **99.43%**, F1 **99.61%**, AUC **99.93%**, sens **99.48%**, spec **99.26%** (3 errors / 523). Fine-tune beats frozen by **+5.48%** acc on average. Ensemble added nothing.
- **Why it matters:** Best controlled “scratch vs TL vs freeze vs fine-tune” study. Also the clearest warning: **99% is on a reshuffled split, not Kermany’s official test.**

**Highlights:** Grad-CAM on TP/TN/FP/FN. Seed 42. Clinical metrics (PPV/NPV).  
**Weaknesses:** Binary only. No external test. Pediatric, one hospital. Numbers not comparable to official-split papers.

---

## Themes And Consensus

1. **Transfer-learning CNN is the default.** Inception V3 (Kermany) → VGG16 (Rajaraman) → ResNet/DenseNet/EfficientNet/Xception. The source paper **froze** the conv stack; later work usually unfreezes it. Fine-tuning the last blocks beats a frozen backbone by ~5 points in the 2026 comparative study — the opposite of Kermany’s 2018 overfitting note. Scratch CNNs still work for binary (Stephen 93.7%, LightPneumoNet 94.2%) but lose to fine-tuned ResNet/DenseNet when the split is held fixed.

2. **DenseNet and ResNet are the workhorse backbones.** DenseNet121/201 win Rahman and Salehi; ResNet50 wins the 2026 bake-off. Kundu’s ensemble of both still posts a high Kermany score, then drops on RSNA.

3. **Imbalance is handled with augmentation, not fancy losses.** Standard toolkit: rotate / shift / shear / zoom / flip; minority oversampling (Ayan); balance-to-N (Rahman 4,500/class); SMOTE (Salehi); class weights (LightPneumoNet); GAN/ADA synthetics (XCCNet). Almost nobody reports focal loss on Kermany (that idea lives in COVID-era 4-class papers such as FLANNEL).

4. **Preprocessing is shallow in early papers, heavy later.** Early: resize + 1/255 or ImageNet norm. Stronger recipes: lung ROI (Rajaraman), CLAHE + segmentation + rib suppression (XCCNet), grayscale (LightPneumoNet). Horizontal flip is not free — some authors disable it because left/right anatomy is meaningful.

5. **Metrics are accuracy, recall, specificity, F1, AUC — not QWK.** Kermany is a 2- or 3-class problem, not an ordinal 5-grade scale. Screening papers optimize **recall**; radiology-journal papers also report specificity and warn about leakage. Three-class accuracy is the honest headline; binary accuracy is inflated by the easy normal-vs-obvious-consolidation split.

6. **Viral pneumonia is the minority-hard class.** Already visible in the source paper: bacterial-vs-viral accuracy **90.7%** / sensitivity **88.6%** vs pneumonia-vs-normal **92.8%** / **93.2%**. Ayan later: viral recall 77% vs bacterial 98%. Rahman: 3-class 93.3% vs binary 98%. Same role as Severe/Proliferative on APTOS.

7. **Paper split vs Kaggle split vs reshuffled split.** Cell protocol: **5,232 train / 624 test**, patients held out. Kaggle’s `5,216 / 16 / 624` is that train set with 16 images moved into a toy val folder. Papers that keep the 624 test report 86–95%. Papers that carve 80/10/10 from train (XCCNet, 2026 comparative) report 99%. **Do not rank those numbers on one leaderboard.**

8. **Kermany does not travel.** Kundu’s 98.8% → 87% on RSNA is the existence proof. Salehi and the 2026 paper both list single-center pediatric data as the main limitation.

## Open Questions And Debates

- **Are 99% models learning pneumonia or site artifacts?** No multi-hospital Kermany-style label protocol. Patient-level leakage is acknowledged (Salehi) but rarely stress-tested.
- **Should the task be binary or 3-class?** Binary is publishable and clinically incomplete: antibiotics depend on bacterial vs viral. 3-class numbers are 4–8 points worse and should be the project default.
- **Freeze vs fine-tune.** Kermany STAR Methods: unfreezing Inception V3 **hurt**. 2026 comparative: fine-tuning last ResNet/DenseNet blocks **+5.48%**. Reproduce both on the official 624 test.
- **Ensemble vs one well-fine-tuned net.** Ayan and Kundu need three towers. The 2026 study found ensembles added **zero** once ResNet50 was fine-tuned. For a semester project, one backbone plus a correct split is the better experiment.
- **Do we need lung segmentation?** Rajaraman and XCCNet say yes; most Kaggle notebooks say no. A clean ablation (full film vs cropped lung) is still worth doing.

## Emerging Trends

| Period | What people did |
| --- | --- |
| 2018–2019 | Inception/VGG, scratch CNNs, first CAM papers |
| 2020–2021 | ImageNet zoo; first honest radiology numbers; first RSNA external test |
| 2024–2026 | Hybrid CNN–Transformer, GAN minority synthesis, class weights, lightweight nets, Grad-CAM/LIME; people finally talk about the 16-image val set |

## Implications For This Project

Treat Kermany the way the example brief treated APTOS:

- **Search query analog:** “pediatric pneumonia chest X-ray deep learning Kermany.”
- **Backbone analog:** ResNet50 or DenseNet121, ImageNet init; **ablate frozen vs last-block fine-tune** (Kermany froze; later papers unfreeze).
- **Imbalance analog:** class weights or minority augmentation; do not only report accuracy.
- **Preprocess analog:** 224×224, ImageNet norm, modest rotation/shift; optional lung crop / CLAHE.
- **Metric analog:** APTOS uses QWK; Kermany uses **accuracy + AUC + macro-F1**, with **per-class recall** for viral.
- **Hard class analog:** Severe/Proliferative DR → **viral pneumonia** (and the normal class when you optimize only recall).

Recommended baseline for the practical part: Cell 624-patient test **or** a documented patient-level 80/10/10, never both silently; binary **and** bacterial-vs-viral (the two tasks the source paper actually reports); ResNet50 frozen vs fine-tune vs a small scratch CNN; Grad-CAM on errors, aimed at the signs in Figure S6 (focal consolidation vs interstitial pattern).

## Sources

### Dataset-defining / foundational

1. Kermany et al., 2018. *Cell*. https://doi.org/10.1016/j.cell.2018.02.010 — local file `kermany_research_paper.pdf`; data https://doi.org/10.17632/rscbjbr9sj.3
2. Rajaraman et al., 2018. *Appl. Sci.* https://doi.org/10.3390/app8101715
3. Stephen et al., 2019. *J. Healthc. Eng.* https://doi.org/10.1155/2019/4180949

### Transfer learning and the hard class

4. Rahman et al., 2020. arXiv:2004.06578
5. Salehi et al., 2021. *Br. J. Radiol.* https://doi.org/10.1259/bjr.20201263
6. Ayan et al., 2021. *Arab. J. Sci. Eng.* https://doi.org/10.1007/s13369-021-06127-z
7. Kundu et al., 2021. *PLoS One*. https://doi.org/10.1371/journal.pone.0256630

### Recent (hybrid, lightweight)

8. XCCNet, 2024. arXiv:2410.16143
9. Chauhan et al., 2025. LightPneumoNet. arXiv:2510.11232
10. Comparative TL vs CNN, 2026. arXiv:2601.00837

### Access notes

Every paper below was read in full. Abstract-only items and extra ImageNet ensembles that only repeat binary accuracy were dropped.
- Local PDF: Kermany 2018, `kermany_research_paper.pdf`.
- Full arXiv text: Rahman 2020, XCCNet 2024, LightPneumoNet 2025, comparative 2026.
- Open PMC/MDPI: Stephen, Rajaraman, Salehi, Ayan, Kundu.

## Rerun Inputs

```
workflow: firecrawl-research-papers
topic: pediatric pneumonia chest X-ray classification on the Kermany / Guangzhou Women and Children’s Medical Center dataset
target_count: 10
output: markdown literature survey + reveal.js html-slides
queries:
  - pediatric chest X-ray pneumonia classification deep learning Kermany Guangzhou
  - Kermany 2018 chest X-ray pneumonia CNN transfer learning ResNet DenseNet EfficientNet
  - viral versus bacterial pneumonia chest radiograph deep learning class imbalance pediatric
seeds:
  - pmid:29474911
  - pmcid:PMC6458916
  - arxiv:2004.06578
  - pmcid:PMC7250407
extraction_fields: [backbone, class_imbalance, preprocessing, metrics]
```
