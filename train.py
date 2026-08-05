import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config import ModelConfig, TrainConfig
from dataset import build_dataset_loaders, load_dti_frame
from metric import classification_metrics
from Model import CSDFDTI
from protocol import make_protocol_split, save_split, validate_protocol
from utils import count_parameters, save_checkpoint, set_seed, write_json


def move_batch(batch, device):
    return batch.to(
        device,
        non_blocking=device.type == "cuda",
    )


def run_training_epoch(model, loader, optimizer, device):
    model.train()
    totals = {
        "loss": [],
        "classification_loss": [],
        "csd_loss": [],
    }
    progress = tqdm(loader, leave=False)
    for batch in progress:
        batch = move_batch(batch, device)
        logits, details = model(batch, return_details=True)
        loss_values = model.loss(logits, batch.labels, details)
        optimizer.zero_grad(set_to_none=True)
        loss_values["loss"].backward()
        optimizer.step()
        for key in totals:
            totals[key].append(float(loss_values[key].detach().cpu()))
        progress.set_postfix(loss=f"{totals['loss'][-1]:.4f}")
    return {
        key: float(np.mean(values)) if values else float("nan")
        for key, values in totals.items()
    }


def collect_predictions(model, loader, device):
    model.eval()
    labels = []
    probabilities = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            probability = model.predict_proba(batch)
            labels.extend(batch.labels.detach().cpu().tolist())
            probabilities.extend(probability.detach().cpu().tolist())
    return np.asarray(labels), np.asarray(probabilities)


def evaluate(model, loader, device, threshold):
    labels, probabilities = collect_predictions(model, loader, device)
    return classification_metrics(labels, probabilities, threshold)


def create_optimizer(model, train_config):
    return torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )


def create_scheduler(optimizer, patience):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(2, patience // 4),
        min_lr=1e-6,
    )


def train_model(
    model,
    loaders,
    optimizer,
    scheduler,
    train_config,
    device,
    output_path,
):
    best_score = -float("inf")
    stale_epochs = 0
    history = []
    for epoch in range(1, train_config.epochs + 1):
        losses = run_training_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
        )
        valid_metrics = evaluate(
            model,
            loaders["valid"],
            device,
            train_config.threshold,
        )
        scheduler.step(valid_metrics["auprc"])
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **losses,
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
        }
        history.append(record)
        print(record)
        if valid_metrics["auprc"] > best_score:
            best_score = valid_metrics["auprc"]
            stale_epochs = 0
            save_checkpoint(
                output_path,
                model,
                optimizer,
                epoch,
                valid_metrics,
                model.config,
                train_config,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= train_config.patience:
            break
    return history


def load_best_model(model, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    return state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="outputs/csdf_dti.pt")
    parser.add_argument("--protocol", choices=("E1", "E2", "E3", "E4"), default="E1")
    parser.add_argument("--max-protein-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_config = ModelConfig()
    train_config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
    )
    set_seed(train_config.seed)
    frame = load_dti_frame(args.data)
    split = make_protocol_split(
        frame,
        protocol=args.protocol,
        seed=train_config.seed,
    )
    report = validate_protocol(split)
    split_dir = output_path.parent / f"{output_path.stem}_{args.protocol}_split"
    save_split(split, split_dir)
    loaders = build_dataset_loaders(
        split.as_dict(),
        batch_size=train_config.batch_size,
        max_protein_length=args.max_protein_length,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSDFDTI(model_config).to(device)
    optimizer = create_optimizer(model, train_config)
    scheduler = create_scheduler(optimizer, train_config.patience)
    history = train_model(
        model,
        loaders,
        optimizer,
        scheduler,
        train_config,
        device,
        output_path,
    )
    checkpoint = load_best_model(model, output_path, device)
    test_metrics = evaluate(
        model,
        loaders["test"],
        device,
        train_config.threshold,
    )
    result = {
        "protocol": args.protocol,
        "parameters": count_parameters(model),
        "best_epoch": checkpoint["epoch"],
        "split_overlap": report,
        "test_metrics": test_metrics,
    }
    write_json(output_path.parent / "training_history.json", history)
    write_json(output_path.parent / "test_metrics.json", result)
    print(result)


if __name__ == "__main__":
    main()
