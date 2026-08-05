from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset


AMINO_ACIDS = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7,
    "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, "O": 13,
    "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, "U": 19,
    "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25,
}

ID_TO_AMINO_ACID = {value: key for key, value in AMINO_ACIDS.items()}


@dataclass
class GraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor

    @property
    def num_nodes(self):
        return int(self.x.size(0))

    @property
    def num_edges(self):
        return int(self.edge_index.size(1))

    def clone(self):
        return GraphData(
            self.x.clone(),
            self.edge_index.clone(),
            self.edge_attr.clone(),
        )


@dataclass
class GraphBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    batch: torch.Tensor
    ptr: torch.Tensor

    @property
    def num_graphs(self):
        return int(self.ptr.numel() - 1)

    @property
    def num_nodes(self):
        return int(self.x.size(0))

    @property
    def num_edges(self):
        return int(self.edge_index.size(1))

    def to(self, device, non_blocking=False):
        return GraphBatch(
            self.x.to(device, non_blocking=non_blocking),
            self.edge_index.to(device, non_blocking=non_blocking),
            self.edge_attr.to(device, non_blocking=non_blocking),
            self.batch.to(device, non_blocking=non_blocking),
            self.ptr.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self):
        return GraphBatch(
            self.x.pin_memory(),
            self.edge_index.pin_memory(),
            self.edge_attr.pin_memory(),
            self.batch.pin_memory(),
            self.ptr.pin_memory(),
        )


@dataclass
class DTISample:
    graph: GraphData
    protein_ids: torch.Tensor
    label: torch.Tensor
    smiles: str
    protein: str


@dataclass
class DTIBatch:
    graph: GraphBatch
    protein_ids: torch.Tensor
    protein_mask: torch.Tensor
    labels: torch.Tensor
    smiles: tuple
    proteins: tuple

    @property
    def batch_size(self):
        return int(self.labels.numel())

    def to(self, device, non_blocking=False):
        return DTIBatch(
            self.graph.to(device, non_blocking=non_blocking),
            self.protein_ids.to(device, non_blocking=non_blocking),
            self.protein_mask.to(device, non_blocking=non_blocking),
            self.labels.to(device, non_blocking=non_blocking),
            self.smiles,
            self.proteins,
        )

    def pin_memory(self):
        return DTIBatch(
            self.graph.pin_memory(),
            self.protein_ids.pin_memory(),
            self.protein_mask.pin_memory(),
            self.labels.pin_memory(),
            self.smiles,
            self.proteins,
        )


class GraphCache:
    def __init__(self):
        self.graphs = {}

    def __len__(self):
        return len(self.graphs)

    def get(self, smiles):
        if smiles not in self.graphs:
            self.graphs[smiles] = smiles_to_graph(smiles)
        return self.graphs[smiles]

    def build(self, smiles_values):
        for smiles in dict.fromkeys(smiles_values):
            self.get(smiles)
        return self


class ProteinCache:
    def __init__(self, max_length):
        self.max_length = int(max_length)
        self.values = {}

    def __len__(self):
        return len(self.values)

    def get(self, sequence):
        sequence = normalize_protein(sequence)
        if sequence not in self.values:
            self.values[sequence] = encode_protein(sequence, self.max_length)
        return self.values[sequence]

    def build(self, sequences):
        for sequence in dict.fromkeys(sequences):
            self.get(sequence)
        return self


def normalize_smiles(smiles):
    value = str(smiles).strip()
    if not value:
        raise ValueError("empty SMILES")
    mol = Chem.MolFromSmiles(value)
    if mol is None:
        raise ValueError(f"invalid SMILES: {value}")
    return Chem.MolToSmiles(mol, canonical=True)


def normalize_protein(sequence):
    value = "".join(str(sequence).split()).upper()
    if not value:
        raise ValueError("empty protein sequence")
    return value


def validate_label(value):
    numeric = float(value)
    if numeric not in (0.0, 1.0):
        raise ValueError(f"label must be 0 or 1: {value}")
    return numeric


def validate_frame(
    frame,
    smiles_column="SMILES",
    protein_column="Protein",
    label_column="Y",
):
    required = {smiles_column, protein_column, label_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    cleaned = frame[[smiles_column, protein_column, label_column]].copy()
    cleaned = cleaned.dropna().reset_index(drop=True)
    cleaned[smiles_column] = cleaned[smiles_column].map(normalize_smiles)
    cleaned[protein_column] = cleaned[protein_column].map(normalize_protein)
    cleaned[label_column] = cleaned[label_column].map(validate_label).astype(np.float32)
    cleaned = cleaned.drop_duplicates(
        subset=[smiles_column, protein_column, label_column],
        keep="first",
    )
    return cleaned.reset_index(drop=True)


def atom_features(atom):
    hybrid = atom.GetHybridization()
    return [
        atom.GetAtomicNum() / 100.0,
        atom.GetDegree() / 4.0,
        atom.GetFormalCharge() / 4.0,
        atom.GetTotalNumHs() / 4.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        atom.GetMass() / 200.0,
        atom.GetTotalValence() / 8.0,
        float(hybrid == Chem.rdchem.HybridizationType.SP),
        float(hybrid == Chem.rdchem.HybridizationType.SP2),
        float(hybrid == Chem.rdchem.HybridizationType.SP3),
    ]


def bond_features(bond):
    bond_type = bond.GetBondType()
    return [
        float(bond_type == Chem.rdchem.BondType.SINGLE),
        float(bond_type == Chem.rdchem.BondType.DOUBLE),
        float(bond_type == Chem.rdchem.BondType.TRIPLE),
        float(bond_type == Chem.rdchem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def smiles_to_graph(smiles):
    canonical = normalize_smiles(smiles)
    mol = Chem.MolFromSmiles(canonical)
    atom_matrix = [atom_features(atom) for atom in mol.GetAtoms()]
    if not atom_matrix:
        raise ValueError(f"SMILES contains no atoms: {canonical}")
    x = torch.tensor(atom_matrix, dtype=torch.float32)
    edges = []
    attributes = []
    for bond in mol.GetBonds():
        source = bond.GetBeginAtomIdx()
        target = bond.GetEndAtomIdx()
        feature = bond_features(bond)
        edges.extend([[source, target], [target, source]])
        attributes.extend([feature, feature])
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(attributes, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float32)
    return GraphData(x, edge_index, edge_attr)


def encode_protein(sequence, max_length):
    sequence = normalize_protein(sequence)
    encoded = np.zeros(int(max_length), dtype=np.int64)
    for index, residue in enumerate(sequence[: int(max_length)]):
        encoded[index] = AMINO_ACIDS.get(residue, AMINO_ACIDS["X"])
    return torch.from_numpy(encoded)


def decode_protein(encoded):
    residues = []
    for value in encoded.tolist():
        if value == 0:
            break
        residues.append(ID_TO_AMINO_ACID.get(int(value), "X"))
    return "".join(residues)


def graph_statistics(graph):
    return {
        "nodes": graph.num_nodes,
        "directed_edges": graph.num_edges,
        "atom_feature_dim": int(graph.x.size(-1)),
        "bond_feature_dim": int(graph.edge_attr.size(-1)),
    }


def frame_statistics(
    frame,
    smiles_column="SMILES",
    protein_column="Protein",
    label_column="Y",
):
    if frame.empty:
        return {
            "pairs": 0,
            "drugs": 0,
            "proteins": 0,
            "positives": 0,
            "negatives": 0,
        }
    positives = int((frame[label_column].astype(float) == 1).sum())
    return {
        "pairs": int(len(frame)),
        "drugs": int(frame[smiles_column].nunique()),
        "proteins": int(frame[protein_column].nunique()),
        "positives": positives,
        "negatives": int(len(frame) - positives),
    }


class DTIDataset(Dataset):
    def __init__(
        self,
        frame,
        smiles_column="SMILES",
        protein_column="Protein",
        label_column="Y",
        max_protein_length=1024,
        graph_cache=None,
        validate=True,
    ):
        if validate:
            frame = validate_frame(
                frame,
                smiles_column,
                protein_column,
                label_column,
            )
        self.frame = frame.reset_index(drop=True)
        self.smiles_column = smiles_column
        self.protein_column = protein_column
        self.label_column = label_column
        self.max_protein_length = int(max_protein_length)
        self.graph_cache = graph_cache or GraphCache()
        self.protein_cache = ProteinCache(self.max_protein_length)
        self.graph_cache.build(self.frame[self.smiles_column].tolist())
        self.protein_cache.build(self.frame[self.protein_column].tolist())

    @classmethod
    def from_csv(cls, path, **kwargs):
        return cls(pd.read_csv(path), **kwargs)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[int(index)]
        smiles = row[self.smiles_column]
        protein = row[self.protein_column]
        graph = self.graph_cache.get(smiles)
        protein_ids = self.protein_cache.get(protein)
        label = torch.tensor(float(row[self.label_column]), dtype=torch.float32)
        return DTISample(graph, protein_ids, label, smiles, protein)

    def statistics(self):
        return frame_statistics(
            self.frame,
            self.smiles_column,
            self.protein_column,
            self.label_column,
        )


def batch_graphs(graphs):
    if not graphs:
        raise ValueError("cannot batch an empty graph list")
    node_values = []
    edge_indices = []
    edge_values = []
    batch_index = []
    ptr = [0]
    offset = 0
    for graph_id, graph in enumerate(graphs):
        node_values.append(graph.x)
        edge_indices.append(graph.edge_index + offset)
        edge_values.append(graph.edge_attr)
        batch_index.append(torch.full((graph.num_nodes,), graph_id, dtype=torch.long))
        offset += graph.num_nodes
        ptr.append(offset)
    return GraphBatch(
        torch.cat(node_values, dim=0),
        torch.cat(edge_indices, dim=1),
        torch.cat(edge_values, dim=0),
        torch.cat(batch_index, dim=0),
        torch.tensor(ptr, dtype=torch.long),
    )


def collate_dti(samples):
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    proteins = torch.stack([sample.protein_ids for sample in samples])
    return DTIBatch(
        graph=batch_graphs([sample.graph for sample in samples]),
        protein_ids=proteins,
        protein_mask=proteins.ne(0),
        labels=torch.stack([sample.label for sample in samples]),
        smiles=tuple(sample.smiles for sample in samples),
        proteins=tuple(sample.protein for sample in samples),
    )


def worker_seed(worker_id):
    base = torch.initial_seed() % (2**32)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


def build_loader(
    dataset,
    batch_size,
    shuffle,
    num_workers=0,
    pin_memory=False,
):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=collate_dti,
        worker_init_fn=worker_seed if num_workers else None,
    )


def build_dataset_loaders(
    split_frames,
    batch_size,
    max_protein_length=1024,
    num_workers=0,
    pin_memory=False,
):
    cache = GraphCache()
    all_smiles = []
    for frame in split_frames.values():
        all_smiles.extend(frame["SMILES"].tolist())
    cache.build(all_smiles)
    datasets = {
        name: DTIDataset(
            frame,
            max_protein_length=max_protein_length,
            graph_cache=cache,
            validate=False,
        )
        for name, frame in split_frames.items()
    }
    return {
        "train": build_loader(
            datasets["train"],
            batch_size,
            True,
            num_workers,
            pin_memory,
        ),
        "valid": build_loader(
            datasets["valid"],
            batch_size,
            False,
            num_workers,
            pin_memory,
        ),
        "test": build_loader(
            datasets["test"],
            batch_size,
            False,
            num_workers,
            pin_memory,
        ),
    }


def load_dti_frame(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in (".tsv", ".txt"):
        frame = pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"unsupported data format: {path.suffix}")
    return validate_frame(frame)
