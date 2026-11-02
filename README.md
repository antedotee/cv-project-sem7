# Pediatric pneumonia — literature survey and XCCNet baseline

Course project on chest X-ray classification. The literature survey and the slide deck cover ten papers. The baseline reproduces the classifier from Raghaw et al., XCCNet (arXiv:2410.16143), on the Kermany Guangzhou pediatric set.

## Layout

- `literature-survey.md` — survey notes
- `presentation/index.html` — slide deck (Reveal.js; open the file in a browser)
- `report/literature_review.tex` — long-form survey
- `report/ieee_literature_survey.tex` — IEEE survey (`IEEEtran.cls` is the class file)
- `baseline/xccnet.py` — model, loss, split, and training loop
- `baseline/xccnet_colab.ipynb` — Colab notebook that trains on a GPU

## Baseline

The notebook is meant to run on a Google Colab GPU runtime. It uses the full Kermany release (1,583 normal and 4,273 pneumonia), pools the Kaggle train/val/test folders, and draws a stratified 80/10/10 split. That is the split in the XCCNet paper, not the official 624-patient test from Kermany et al., *Cell* 2018.

Open `baseline/xccnet_colab.ipynb` in Colab, set the runtime to GPU, and provide a Kaggle API token so the notebook can download [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). The notebook writes metrics to `baseline/results/metrics.json`.

## Hypothesis and ablations (M3–M5)

The hypothesis and predictions were committed before the code change (`milestones/M3-hypothesis.md`). We replace cross-entropy with focal loss (γ = 2) and add two controls: a threshold tuned on validation (A′) and the baseline without the lung mask (C). Each row is trained with three seeds on the same split.

| Row | Change | Accuracy | Sensitivity | Specificity | AUC |
| --- | --- | --- | --- | --- | --- |
| A | Baseline | 92.71 ± 0.55 | 95.32 ± 1.87 | 85.65 ± 3.25 | 0.978 |
| A′ | Threshold tuned on val | 92.42 ± 0.69 | 93.29 ± 1.51 | 90.08 ± 2.22 | 0.978 |
| B | Focal loss γ = 2 | 91.68 ± 0.39 | 96.33 ± 1.08 | 79.11 ± 2.53 | 0.975 |
| C | Lung mask off | 95.78 ± 0.26 | 97.97 ± 0.36 | 89.87 ± 1.90 | 0.993 |

- `baseline/run_experiment.py` trains and tests one row on a GPU, for example `python run_experiment.py --name B_focal_g2 --loss focal --gamma 2`.
- `experiments/summarize.py` rebuilds the table and figures from the saved per-film probabilities.
- `milestones/` holds the checklist (M1), hypothesis (M3), results (M4), failure analysis (M5), and a defence guide.
- `report/xccnet_paper.tex` is the IEEE paper (`xccnet_paper.pdf`).
