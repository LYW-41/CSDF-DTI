# CSDF-DTI

This repository provides the implementation of a shortcut-aware deep learning framework for drug–target interaction prediction.

---

## Environment Setup

### Requirements

- Python 3.10
- PyTorch 2.7.1
- CUDA 12.8
- RDKit 2023.9.6

### Main Dependencies

```text
numpy==1.26.4
pandas==2.3.3
scipy==1.15.3
scikit-learn==1.7.2
tqdm==4.67.1
PyYAML==6.0.3
matplotlib==3.10.8
joblib==1.5.3
```

## Run

```bash
python train.py --data data/Human/sample.csv --protocol E1
```
