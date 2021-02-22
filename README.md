Uncertainty calibration for label histograms
===

This repository provides methods for calibrating probability predictions on classification and inter-rater disagreement using label histograms, i.e., multiple labels per instance.

## Setup

```
pip install -r requirements.txt
```

## Examples

Prepareation of neural net classifiers for example data with label uncertainty.
```bash
make train_mixed_{mnist,cifar10}
make pred_mixed_{mnist,cfiar10}
```

Applying temperature scaling and alpha-calibration on the trained models.
```bash
make calib_mixed_p1s
make calib_mixed_p2s
```

## Reference

* Takahiro Mimori, Keiko Sasada, Hirotaka Matsui, Issei Sato. Diagnostic Uncertainty Calibration: Towards Reliable Machine Predictions in Medical Domain. In AISTATS, 2021. [[arXiv]](https://arxiv.org/abs/2007.01659)
