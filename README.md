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
