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
    candidate, train = stratified_rows(frame, 0.2, seed, label_column)
    valid, test = split_validation_test(candidate, seed + 1, label_column)
    return SplitFrames(train, valid, test, "E1")


def sample_entities(values, fraction, seed):
    unique = np.array(sorted(set(values)))
    if unique.size < 2:
        raise ValueError("cold-start splitting requires at least two entities")
    generator = np.random.default_rng(seed)
    generator.shuffle(unique)
    count = max(1, int(round(unique.size * fraction)))
    count = min(count, unique.size - 1)
    return set(unique[:count].tolist())


def protocol_masks(
    frame,
    protocol,
    unseen_drugs,
    unseen_proteins,
    smiles_column,
    protein_column,
):
    drug_unseen = frame[smiles_column].isin(unseen_drugs)
    protein_unseen = frame[protein_column].isin(unseen_proteins)
    if protocol == "E2":
        return drug_unseen, ~drug_unseen
    if protocol == "E3":
        return protein_unseen, ~protein_unseen
    candidate = drug_unseen & protein_unseen
    training = ~drug_unseen & ~protein_unseen
    return candidate, training


def entity_protocol_split(
    frame,
    protocol,
    seed=2026,
    smiles_column="SMILES",
    protein_column="Protein",
    label_column="Y",
    entity_fraction=0.2,
    max_attempts=100,
):
    if protocol not in ("E2", "E3", "E4"):
        raise ValueError(f"unsupported entity protocol: {protocol}")
    for attempt in range(max_attempts):
        current_seed = seed + attempt
        unseen_drugs = set()
        unseen_proteins = set()
        if protocol in ("E2", "E4"):
            unseen_drugs = sample_entities(
                frame[smiles_column],
                entity_fraction,
                current_seed,
            )
        if protocol in ("E3", "E4"):
            unseen_proteins = sample_entities(
                frame[protein_column],
                entity_fraction,
                current_seed + 1009,
            )
        candidate_mask, train_mask = protocol_masks(
            frame,
            protocol,
            unseen_drugs,
            unseen_proteins,
            smiles_column,
            protein_column,
        )
        candidate = frame.loc[candidate_mask].reset_index(drop=True)
        train = frame.loc[train_mask].reset_index(drop=True)
        if candidate.empty or train.empty:
            continue
        if candidate[label_column].nunique() != 2:
            continue
        if train[label_column].nunique() != 2:
            continue
        valid, test = split_validation_test(candidate, current_seed + 17, label_column)
        if valid.empty or test.empty:
            continue
        return SplitFrames(train, valid, test, protocol)
    raise RuntimeError(f"unable to construct {protocol} split")


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
    if split.protocol == "E2" and report["test_drug_overlap"] != 0:
        raise ValueError("E2 test drugs overlap with training drugs")
    if split.protocol == "E3" and report["test_protein_overlap"] != 0:
        raise ValueError("E3 test proteins overlap with training proteins")
    if split.protocol == "E4":
        if report["test_drug_overlap"] != 0:
            raise ValueError("E4 test drugs overlap with training drugs")
        if report["test_protein_overlap"] != 0:
            raise ValueError("E4 test proteins overlap with training proteins")
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
