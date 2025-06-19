import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_stage1 import Stage1FAUModel
from data.datasets import get_dataset
from utils.metrics import compute_f1_score, compute_accuracy
from utils.logger import Logger


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    for video, labels in tqdm(dataloader, desc="Training"):
        video, labels = video.to(device), labels.to(device)

        # Fake edge_index (temporal chain) — replace with real graph later
        B, T, _, _, _ = video.shape
        total_nodes = B * T
        edge_index = torch.tensor([[i, i+1] for i in range(total_nodes - 1)] +
                                  [[i+1, i] for i in range(total_nodes - 1)], dtype=torch.long).t().to(device)

        # Forward pass
        fau_logits, _ = model(video, edge_index)
        loss = criterion(fau_logits.view(-1, fau_logits.size(-1)), labels.repeat_interleave(T, dim=0))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = (fau_logits.view(-1, fau_logits.size(-1)) > 0.5).float().cpu()
        all_preds.append(preds)
        all_targets.append(labels.repeat_interleave(T, dim=0).cpu())

    avg_loss = total_loss / len(dataloader)
    f1 = compute_f1_score(torch.cat(all_preds), torch.cat(all_targets))
    acc = compute_accuracy(torch.cat(all_preds), torch.cat(all_targets))

    return avg_loss, f1, acc


def train_stage1(config_path):
    config = load_config(config_path)
    stage1_cfg = config['stage1']
    data_cfg = config['data']
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Logger
    logger = Logger(log_file=os.path.join(config['log_dir'], "train_stage1.log"))

    # Dataset
    train_set = get_dataset(
        name=data_cfg["dataset_name"],
        split_csv=data_cfg["label_file"],
        root_dir=data_cfg["dataset_root"],
        clip_length=data_cfg["clip_length"],
        label_type="fau"
    )
    train_loader = DataLoader(train_set, batch_size=data_cfg["batch_size"],
                              shuffle=data_cfg["shuffle"], num_workers=data_cfg["num_workers"])

    # Model
    model = Stage1FAUModel(
        video_mae_name=stage1_cfg["video_mae_name"],
        num_faus=stage1_cfg["num_faus"],
        gnn_out_dim=stage1_cfg["gnn_out_dim"],
        freeze_video_mae=stage1_cfg["freeze_video_mae"]
    ).to(device)

    # Optimizer & loss
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=stage1_cfg["learning_rate"],
                                  weight_decay=stage1_cfg["weight_decay"])
    criterion = nn.BCELoss()

    # Training loop
    for epoch in range(stage1_cfg["num_epochs"]):
        loss, f1, acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        logger.log(f"Epoch {epoch+1}/{stage1_cfg['num_epochs']} | Loss: {loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0 or (epoch + 1) == stage1_cfg['num_epochs']:
            torch.save(model.state_dict(), os.path.join(config["checkpoint_dir"], f"stage1_epoch{epoch+1}.pt"))

    # Final save
    torch.save(model.state_dict(), stage1_cfg["save_path"])
    logger.log(f"✅ Training complete. Final model saved to {stage1_cfg['save_path']}")


if __name__ == "__main__":
    train_stage1("config/config.yaml")
