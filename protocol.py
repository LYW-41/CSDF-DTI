from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from dataset import frame_statistics, validate_frame


SUPPORTED_PROTOCOLS = ("E1", "E2", "E3", "E4")


@dataclass
class SplitFrames:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    protocol: str

    def as_dict(self):
        return {
            "train": self.train,
            "valid": self.valid,
            "test": self.test,
        }

    def summary(self):
        return {
            "protocol": self.protocol,
            "train": frame_statistics(self.train),
            "valid": frame_statistics(self.valid),
            "test": frame_statistics(self.test),
            "overlap": split_overlap_report(self),
        }


def stratified_rows(frame, fraction, seed, label_column="Y"):
    generator = np.random.default_rng(seed)
    selected = []
    remaining = []
    for _, group in frame.groupby(label_column, sort=False):
        indices = group.index.to_numpy().copy()
        generator.shuffle(indices)
        count = max(1, int(round(len(indices) * fraction)))
        if len(indices) > 1:
            count = min(count, len(indices) - 1)
        selected.extend(indices[:count].tolist())
        remaining.extend(indices[count:].tolist())
    first = frame.loc[selected].sample(frac=1.0, random_state=seed)
    second = frame.loc[remaining].sample(frac=1.0, random_state=seed)
    return first.reset_index(drop=True), second.reset_index(drop=True)


def split_validation_test(frame, seed, label_column="Y"):
    test, valid = stratified_rows(frame, 2.0 / 3.0, seed, label_column)
    return valid, test


def random_protocol_split(frame, seed=2026, label_column="Y"):
    candidate, train = stratified_rows(frame, 0.3, seed, label_column)
    valid, test = split_validation_test(candidate, seed + 1, label_column)
    return SplitFrames(train, valid, test, "E1")

def entity_protocol_split(
    frame,
    protocol,
    seed=2026,
    smiles_column="SMILES",
    protein_column="Protein",
    label_column="Y",
    max_attempts=100,
):
    if protocol not in ("E2", "E3", "E4"):
        raise ValueError(f"unsupported entity protocol: {protocol}")

    for attempt in range(max_attempts):
        current_seed = seed + attempt

        # Randomly select 20% of interaction pairs as candidate samples.
        candidate, train = stratified_rows(
            frame,
            0.2,
            current_seed,
            label_column,
        )

        train_drugs = set(train[smiles_column])
        train_proteins = set(train[protein_column])

        drug_seen = candidate[smiles_column].isin(train_drugs)
        protein_seen = candidate[protein_column].isin(train_proteins)

        if protocol == "E2":
            # Unseen drugs and known proteins.
            mask = (~drug_seen) & protein_seen

        elif protocol == "E3":
            # Known drugs and unseen proteins.
            mask = drug_seen & (~protein_seen)

        else:  # E4
            # Both drugs and proteins are unseen.
            mask = (~drug_seen) & (~protein_seen)

        retained = candidate.loc[mask].reset_index(drop=True)

        if retained.empty:
            continue

        if retained[label_column].nunique() != 2:
            continue

        # Validation:test = 1:2.
        valid, test = split_validation_test(
            retained,
            current_seed + 17,
            label_column,
        )

        if valid.empty or test.empty:
            continue

        if valid[label_column].nunique() != 2:
            continue

        if test[label_column].nunique() != 2:
            continue

        return SplitFrames(
            train.reset_index(drop=True),
            valid.reset_index(drop=True),
            test.reset_index(drop=True),
            protocol,
        )

    raise RuntimeError(
        f"unable to construct {protocol} split after {max_attempts} attempts"
    )

def make_protocol_split(
    frame,
    protocol="E1",
    seed=2026,
    smiles_column="SMILES",
    protein_column="Protein",
    label_column="Y",
):
    protocol = protocol.upper()
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"protocol must be one of {SUPPORTED_PROTOCOLS}")
    frame = validate_frame(
        frame,
        smiles_column,
        protein_column,
        label_column,
    )
    if protocol == "E1":
        return random_protocol_split(frame, seed, label_column)
    return entity_protocol_split(
        frame,
        protocol,
        seed,
        smiles_column,
        protein_column,
        label_column,
    )


def split_overlap_report(
    split,
    smiles_column="SMILES",
    protein_column="Protein",
):
    train_drugs = set(split.train[smiles_column])
    train_proteins = set(split.train[protein_column])
    valid_drugs = set(split.valid[smiles_column])
    valid_proteins = set(split.valid[protein_column])
    test_drugs = set(split.test[smiles_column])
    test_proteins = set(split.test[protein_column])
    return {
        "protocol": split.protocol,
        "valid_drug_overlap": len(train_drugs & valid_drugs),
        "valid_protein_overlap": len(train_proteins & valid_proteins),
        "test_drug_overlap": len(train_drugs & test_drugs),
        "test_protein_overlap": len(train_proteins & test_proteins),
        "train_drugs": len(train_drugs),
        "train_proteins": len(train_proteins),
        "valid_drugs": len(valid_drugs),
        "valid_proteins": len(valid_proteins),
        "test_drugs": len(test_drugs),
        "test_proteins": len(test_proteins),
    }


def validate_protocol(split):
    report = split_overlap_report(split)

    if split.protocol == "E2":
        if report["valid_drug_overlap"] != 0:
            raise ValueError(
                "E2 validation drugs overlap with training drugs"
            )

        if report["test_drug_overlap"] != 0:
            raise ValueError(
                "E2 test drugs overlap with training drugs"
            )

        if report["valid_protein_overlap"] != report["valid_proteins"]:
            raise ValueError(
                "E2 validation contains proteins absent from training"
            )

        if report["test_protein_overlap"] != report["test_proteins"]:
            raise ValueError(
                "E2 test contains proteins absent from training"
            )

    elif split.protocol == "E3":
        if report["valid_protein_overlap"] != 0:
            raise ValueError(
                "E3 validation proteins overlap with training proteins"
            )

        if report["test_protein_overlap"] != 0:
            raise ValueError(
                "E3 test proteins overlap with training proteins"
            )

        if report["valid_drug_overlap"] != report["valid_drugs"]:
            raise ValueError(
                "E3 validation contains drugs absent from training"
            )

        if report["test_drug_overlap"] != report["test_drugs"]:
            raise ValueError(
                "E3 test contains drugs absent from training"
            )

    elif split.protocol == "E4":
        if report["valid_drug_overlap"] != 0:
            raise ValueError(
                "E4 validation drugs overlap with training drugs"
            )

        if report["test_drug_overlap"] != 0:
            raise ValueError(
                "E4 test drugs overlap with training drugs"
            )

        if report["valid_protein_overlap"] != 0:
            raise ValueError(
                "E4 validation proteins overlap with training proteins"
            )

        if report["test_protein_overlap"] != 0:
            raise ValueError(
                "E4 test proteins overlap with training proteins"
            )

    return report


def save_split(split, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in split.as_dict().items():
        frame.to_csv(directory / f"{name}.csv", index=False)
    with open(directory / "split.json", "w", encoding="utf-8") as handle:
        json.dump(split.summary(), handle, indent=2)


def load_split(directory):
    directory = Path(directory)
    with open(directory / "split.json", "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    split = SplitFrames(
        train=pd.read_csv(directory / "train.csv"),
        valid=pd.read_csv(directory / "valid.csv"),
        test=pd.read_csv(directory / "test.csv"),
        protocol=metadata["protocol"],
    )
    validate_protocol(split)
    return split
