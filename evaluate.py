import argparse
from pathlib import Path

import torch

from config import ModelConfig, TrainConfig
from dataset import build_dataset_loaders, load_dti_frame
from metric import classification_metrics
from Model import CSDFDTI
from protocol import make_protocol_split, validate_protocol
from utils import set_seed, write_json


def move_batch(batch, device):
    return batch.to(
        device,
        non_blocking=device.type == "cuda",
    )


def collect_predictions(model, loader, device):
    model.eval()

    labels = []
    probabilities = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)

            probability = model.predict_proba(batch)

            labels.extend(batch.labels.detach().cpu().tolist())
            probabilities.extend(
                probability.detach().cpu().tolist()
            )

    return labels, probabilities


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained CSDF-DTI checkpoint."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to the DTI dataset CSV file.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the pretrained .pth checkpoint.",
    )

    parser.add_argument(
        "--protocol",
        choices=("E1", "E2", "E3", "E4"),
        default="E2",
        help="Evaluation protocol.",
    )

    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="Path for saving evaluation results.",
    )

    parser.add_argument(
        "--max-protein-length",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Build the same data partition used during training.
    frame = load_dti_frame(args.data)

    split = make_protocol_split(
        frame,
        protocol=args.protocol,
        seed=args.seed,
    )

    report = validate_protocol(split)

    loaders = build_dataset_loaders(
        split.as_dict(),
        batch_size=args.batch_size,
        max_protein_length=args.max_protein_length,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Build CSDF-DTI using the same model configuration.
    model_config = ModelConfig()

    model = CSDFDTI(model_config).to(device)

    # Load pretrained checkpoint.
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model"])

    # Use the same classification threshold as training.
    train_config = TrainConfig()
    threshold = train_config.threshold

    labels, probabilities = collect_predictions(
        model,
        loaders["test"],
        device,
    )

    test_metrics = classification_metrics(
        labels,
        probabilities,
        threshold,
    )

    result = {
        "protocol": args.protocol,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split_overlap": report,
        "test_metrics": test_metrics,
    }

    output_path = Path(args.output)

    if output_path.parent != Path("."):
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    write_json(output_path, result)

    print(result)


if __name__ == "__main__":
    main()
