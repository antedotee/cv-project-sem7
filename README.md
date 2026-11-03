# Pediatric Pneumonia from Chest X-rays: Reproducing XCCNet and Testing One Change

Semester 7 Computer Vision project by Kartik Yadav, Gangadhar Yadav, and Sahil Sarawgi.

We picked a published method for classifying children's chest X-rays as **normal** or **pneumonia**, rebuilt it, reproduced its result, and then tested one change of our own. We wrote the hypothesis and prediction before writing the code for it. The final write-up is an IEEE-format paper: [`report/research_report.pdf`](report/research_report.pdf).

## The project in short

- **Topic:** pneumonia detection in pediatric chest X-rays.
- **Dataset:** Kermany et al. (2018), Guangzhou Women and Children's Medical Center. 5,856 films: 1,583 normal and 4,273 pneumonia. We used the [Kaggle mirror](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
- **Base paper:** Raghaw et al., *XCCNet*, arXiv:2410.16143 (2024). It joins a dilated CNN with a frozen CLIP image encoder and trains both with a contrastive loss plus cross-entropy.
- **Our change:** replace cross-entropy with **focal loss** (γ = 2).
- **Result:** our hypothesis was wrong. Focal loss lowered specificity and did not change AUC. A second experiment found that the lung mask in the pipeline hurts on children's films.

## What we did, step by step

| Step | What we did | Where it is |
| --- | --- | --- |
| 1. Literature survey | Read ten papers (2018–2026) on this dataset. Compared methods, splits, and metrics. | `report/ieee_literature_survey.tex`, `report/literature_review.tex` |
| 2. Base paper | Chose XCCNet. It has no public code, so we rebuilt the model from the paper. | `baseline/xccnet.py` |
| 3. Reproduction | Trained on the paper's stratified 80/10/10 split on a free Colab GPU. First run: 94.87% test accuracy. | `baseline/xccnet_colab.ipynb`, `baseline/results/metrics.json` |
| 4. Hypothesis | Predicted that focal loss would raise specificity from 85% to 88–91% with AUC unchanged. Committed on the slides before the code change. | `presentation/` (Hypothesis and Predictions slides) |
| 5. Experiments | Four configurations, three random seeds each, same split for every run. | `baseline/run_experiment.py`, `experiments/` |
| 6. Analysis and paper | Confusion matrices, ROC curves, misclassified films with Grad-CAM, reproduction gap, limitations. | `report/research_report.pdf`, `presentation/presentation.pdf` |

### The model

1. Each X-ray is resized to 256×256 and contrast-enhanced with CLAHE.
2. A pretrained lung-segmentation model (ChestX-Det PSPNet from torchxrayvision) blacks out everything outside the lungs.
3. Two branches read the image:
   - a five-block dilated CNN, trained from scratch, that looks at local texture (2,304 features);
   - a frozen CLIP ViT-B/32 that looks at the whole image (512 features).
4. The two feature vectors are joined and a small two-layer head outputs the probability of pneumonia.
5. The loss has a contrastive part (the two branches should agree on the same image) and a classification part. The contrastive weight shrinks each epoch.

Two stages of the original paper are not included. The paper does not give enough detail to build its GAN, and its rib-suppression network needs paired bone-suppressed images that this dataset does not have.

## Results

Test set: 585 films (427 pneumonia, 158 normal). Mean ± standard deviation over three seeds. Sensitivity is the share of pneumonia films caught. Specificity is the share of normal films correctly called normal.

| Row | Configuration | Accuracy % | Sensitivity % | Specificity % | AUC |
| --- | --- | --- | --- | --- | --- |
| A | Baseline (cross-entropy) | 92.71 ± 0.55 | 95.32 ± 1.87 | 85.65 ± 3.25 | 0.978 ± 0.004 |
| A′ | Baseline, decision threshold tuned on validation | 92.42 ± 0.69 | 93.29 ± 1.51 | 90.08 ± 2.22 | 0.978 ± 0.004 |
| B | Focal loss, γ = 2 (our hypothesis) | 91.68 ± 0.39 | 96.33 ± 1.08 | 79.11 ± 2.53 | 0.975 ± 0.003 |
| C | Baseline without the lung mask | 95.78 ± 0.26 | 97.97 ± 0.36 | 89.87 ± 1.90 | 0.993 ± 0.001 |

### What we learned

- **Focal loss did not help.** It pulled the model's scores toward the middle, so more normal films crossed the 0.5 cut-off and specificity fell from 85.7% to 79.1%. AUC did not change, so the model still ranked the films the same way. With the threshold tuned on validation, both losses reach the same specificity. Row A′ is the control that shows this. The training sampler already balances the two classes, which left focal loss little to fix. We had listed this risk before running.
- **The lung mask hurts.** We predicted that removing it would lower accuracy. Accuracy went up by 3.1 points instead. The mask model was trained on adult X-rays and often cuts off parts of a child's lung. Normal films with the smallest masks had a 23.4% error rate, against 14.4% for the rest. Without the mask, though, the model may also use shortcuts such as the "R" marker, which this dataset alone cannot rule out.
- **Most errors are normal films called pneumonia.** Among pneumonia films, viral cases are missed more often than bacterial ones (93.4% vs 96.5% recall in row A), which matches earlier papers.
- **Reproduction gap.** The paper reports 97.18% for its setting without the GAN. Its own precision and recall give 94.84% on a test set with our class mix, and our first run reached 94.87%. Our later runs use a fixed file order for the split and an average over three seeds, which gives 92.71%.

## Run records

The experiments ran on a free Colab T4 GPU through the Colab command-line tool, not inside a notebook. Their record is in `experiments/`:

- **`experiments/results.ipynb`** opens all 9 saved runs. It shows each run's settings and its epoch-by-epoch training log, then recomputes the results table, confusion matrices, ROC curves, and failure analysis from the per-film scores. It also checks that the recomputed numbers match the ones each run saved. It needs no GPU and runs in a few seconds.
- **`experiments/colab_logs/session1.ipynb`** and **`session2.ipynb`** are the Colab tool's own history of the two GPU sessions on 2 November 2026: every piece of code sent to the GPU, what it printed (including the Tesla T4 device line, training epochs, and each run's final `DONE` line), and every result file downloaded, all with timestamps. The first session was dropped by Colab partway through, so the second one restarts the remaining runs.

## Repository structure

```
baseline/
  xccnet.py              Model, losses (cross-entropy or focal), metrics, data split, training loop
  run_experiment.py      Trains and tests one configuration; saves metrics and per-film scores
  xccnet_colab.ipynb     Original Colab notebook for the first baseline run
  requirements.txt       Extra packages needed on Colab
  checkpoints/           Weights from the first baseline run (94.87%)
  results/               Metrics and figures from the first baseline run
experiments/
  A_baseline*/           Row A, seeds 42, 1, 2
  B_focal_g2*/           Row B, seeds 42, 1, 2
  C_no_mask*/            Row C, seeds 42, 1, 2
                         Each run folder has metrics.json, test_predictions.csv, val_predictions.csv
  results.ipynb          Training logs of all runs and every number in the paper, recomputed
  colab_logs/            Timestamped history of the two Colab GPU sessions
  summarize.py           Builds the results table, ROC curves, and confusion matrices from the runs
  mask_coverage.csv      Share of each test image kept by the lung mask
  test_files.csv         Test films with their original names and pneumonia subtype
presentation/
  index.html             Slide deck (Reveal.js; open in a browser)
  presentation.pdf       PDF export of the slides
report/
  research_report.tex/.pdf  Final IEEE-format paper
  ieee_literature_survey.tex   Literature survey in IEEE format
  literature_review.tex  Longer literature review
  figures/               Figures used in the paper
  IEEEtran.cls           IEEE LaTeX class
```

## How to run

**Train one configuration (Google Colab, GPU runtime).** Copy `baseline/xccnet.py` and `baseline/run_experiment.py` to the VM and install the extra packages:

```
pip install open_clip_torch torchxrayvision kagglehub
python run_experiment.py --name A_baseline
python run_experiment.py --name B_focal_g2 --loss focal --gamma 2
python run_experiment.py --name C_no_mask --no-lung-mask
```

Add `--seed 1` or `--seed 2` for the other seeds. The data split always uses seed 42, so every run sees the same test films. Each run writes to `/content/results/<name>/` and takes 3–10 minutes on a T4 once the images are preprocessed. The first preprocessing pass takes about 15 minutes.

**Rebuild the table and figures (laptop).** Copy the run folders into `experiments/`, then:

```
pip install numpy scikit-learn matplotlib
python experiments/summarize.py
```

**Compile the paper.**

```
cd report && pdflatex research_report.tex && pdflatex research_report.tex
```

## Limitations

- All films come from one hospital, and the children are aged 1 to 5. There is no test on another hospital's films.
- We use the paper's pooled 80/10/10 split, not the official 624-patient test set. The split is by film, not by child: the 427 pneumonia test films come from 373 children, so some children have films in both training and test. This can raise every score.
- Three seeds on one split. Specificity varies by up to 6 points between seeds.
- The lung mask is a stand-in for the paper's ResUNet++, and the GAN and rib suppression are not reproduced.
- **This is a course research project. It makes no clinical claim.**

## References

- D. S. Kermany et al., "Identifying medical diagnoses and treatable diseases by image-based deep learning," *Cell*, 2018.
- C. S. Raghaw, P. S. Bhore, M. Z. U. Rehman, and N. Kumar, "An explainable contrastive-based dilated convolutional network with transformer for pediatric pneumonia detection," arXiv:2410.16143, 2024.
- T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," ICCV, 2017.
- J. P. Cohen et al., "TorchXRayVision: A library of chest X-ray datasets and models," MIDL, 2022.

The full reference list for the survey and the paper is in `report/`.
