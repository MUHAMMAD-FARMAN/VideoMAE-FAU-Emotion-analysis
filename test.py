import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_stage1 import Stage1FAUModel
from model_stage2 import Stage2EmotionModel
from data.datasets import get_dataset
from utils.metrics import compute_accuracy, compute_f1_score, compute_ccc


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_checkpoint(model, checkpoint_path, device):
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
    else:
        print("⚠️ No valid checkpoint provided. Model will run with random weights.")
    return model


def evaluate_stage1(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for video, labels in tqdm(dataloader, desc="Evaluating Stage 1"):
            video = video.to(device)
            labels = labels.to(device)

            B, T, _, _, _ = video.shape
            total_nodes = B * T
            edge_index = torch.tensor([[i, i + 1] for i in range(total_nodes - 1)] +
                                      [[i + 1, i] for i in range(total_nodes - 1)], dtype=torch.long).t().to(device)

            fau_logits, _ = model(video, edge_index)
            preds = (fau_logits.view(-1, fau_logits.size(-1)) > 0.5).float().cpu()
            all_preds.append(preds)
            all_targets.append(labels.repeat_interleave(T, dim=0).cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    acc = compute_accuracy(preds, targets)
    f1 = compute_f1_score(preds, targets)
    return acc, f1


def evaluate_stage2(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for video, labels in tqdm(dataloader, desc="Evaluating Stage 2"):
            video = video.to(device)
            labels = labels.to(device)

            B, T, _, _, _ = video.shape
            edge_index = torch.tensor([[i, i + 1] for i in range(B*T - 1)] +
                                      [[i + 1, i] for i in range(B*T - 1)], dtype=torch.long).t().to(device)

            logits = model(video, edge_index)
            preds = torch.argmax(logits, dim=-1).view(-1).cpu()
            targets = torch.argmax(labels, dim=-1).repeat_interleave(T).cpu()

            all_preds.append(preds)
            all_targets.append(targets)

    preds = torch.cat(all_preds).float()
    targets = torch.cat(all_targets).float()

    acc = compute_accuracy(preds, targets)
    ccc = compute_ccc(preds, targets)
    return acc, ccc


def run_test(config_path):
    config = load_config(config_path)
    data_cfg = config["data"]
    test_cfg = config["test"]
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = get_dataset(
        name=data_cfg["dataset_name"],
        split_csv=test_cfg["test_split_file"],
        root_dir=data_cfg["dataset_root"],
        clip_length=data_cfg["clip_length"],
        label_type="fau" if test_cfg["stage"] == 1 else "emotion"
    )

    dataloader = DataLoader(dataset,
                            batch_size=data_cfg["batch_size"],
                            shuffle=False,
                            num_workers=data_cfg["num_workers"])

    # Stage 1
    if test_cfg["stage"] == 1:
        model = Stage1FAUModel(
            video_mae_name=config['stage1']['video_mae_name'],
            num_faus=config['stage1']['num_faus'],
            gnn_out_dim=config['stage1']['gnn_out_dim'],
            freeze_video_mae=True
        ).to(device)

        model = load_checkpoint(model, config['stage1']['save_path'], device)
        acc, f1 = evaluate_stage1(model, dataloader, device)
        print(f"\n✅ Stage 1 Evaluation — F1: {f1:.4f} | Accuracy: {acc:.4f}")

    # Stage 2
    elif test_cfg["stage"] == 2:
        model = Stage2EmotionModel(
            video_mae_name=config['stage2']['video_mae_name'],
            num_faus=config['stage2']['num_faus'],
            gnn_out_dim=config['stage2']['gnn_out_dim'],
            num_emotions=config['stage2']['num_emotions'],
            freeze_stage1=True
        ).to(device)

        # Load Stage 1 weights before loading stage 2
        model.stage1_model.load_state_dict(torch.load(config['stage1']['save_path'], map_location=device))
        model = load_checkpoint(model, config['stage2']['save_path'], device)

        acc, ccc = evaluate_stage2(model, dataloader, device)
        print(f"\n✅ Stage 2 Evaluation — Accuracy: {acc:.4f} | CCC: {ccc:.4f}")

    else:
        raise ValueError("Invalid stage number. Choose 1 or 2.")


if __name__ == "__main__":
    run_test("config/config.yaml")
