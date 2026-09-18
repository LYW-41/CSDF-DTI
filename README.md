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
torch==2.7.1
torch-geometric==2.7.0
mamba-ssm==2.2.6.post3
causal-conv1d==1.5.3.post1
einops==0.8.1
rdkit==2023.9.6
aaindex==1.1.2

numpy==1.26.4
pandas==2.3.3
scipy==1.15.3
scikit-learn==1.7.2
tqdm==4.67.1
PyYAML==6.0.3
matplotlib==3.10.8
joblib==1.5.3
```

## Data and Evaluation Protocols

The benchmark datasets used in the experiments are provided in `data.zip`. The evaluation protocols are implemented in `protocol.py`.

For E1, interaction pairs are randomly split into training, validation, and test sets at a ratio of 7:1:2. For E2, E3, and E4, 20% of the interaction
pairs are first selected as candidate evaluation samples, while the remaining 80% form the initial training set. E2 retains pairs with unseen drugs and known proteins, E3 retains pairs with known drugs and unseen proteins, and E4 retains pairs in which both entities are unseen. The retained samples are divided into
validation and test sets at a ratio of 1:2.

All reported results are based on five independent runs.

## Run

Example for the Human dataset:

```bash
python train.py --data data/Human/sample.csv --protocol E1
```

Replace `E1` with `E2`, `E3`, or `E4` to run the corresponding cold start evaluation. The checkpoint with the highest validation AUROC is used for final test evaluation.

### Pretrained checkpoint

The pretrained checkpoint can be downloaded from the [Releases page](https://github.com/LYW-41/CSDF-DTI/releases).

Checkpoint:
- Human dataset, E2 cold-start protocol: human_e2_seed42.pth
