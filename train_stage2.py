import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_stage2 import Stage2EmotionModel
from data.datasets import get_dataset
from utils.metrics import compute_accuracy, compute_ccc
from utils.logger import Logger


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    for video, labels in tqdm(dataloader, desc="Training"):
        video = video.to(device)
        labels = labels.to(device)

        B, T, C, H, W = video.shape
        edge_index = torch.tensor([[i, i+1] for i in range(B*T - 1)] +
                                  [[i+1, i] for i in range(B*T - 1)], dtype=torch.long).t().to(device)

        logits = model(video, edge_index)  # (B, T, num_emotions)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.repeat_interleave(T, dim=0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=-1).view(-1).cpu()
        targets = torch.argmax(labels, dim=-1).repeat_interleave(T).cpu()
        all_preds.append(preds)
        all_targets.append(targets)

    avg_loss = total_loss / len(dataloader)
    acc = compute_accuracy(torch.cat(all_preds), torch.cat(all_targets))
    ccc = compute_ccc(torch.cat(all_preds).float(), torch.cat(all_targets).float())

    return avg_loss, acc, ccc


def train_stage2(config_path):
    config = load_config(config_path)
    data_cfg = config['data']
    stage2_cfg = config['stage2']
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    logger = Logger(log_file=os.path.join(config['log_dir'], "train_stage2.log"))

    # Load dataset
    dataset = get_dataset(
        name=stage2_cfg['emotion_dataset'],
        split_csv=data_cfg['label_file'],
        root_dir=data_cfg['dataset_root'],
        clip_length=data_cfg['clip_length'],
        label_type="emotion"
    )
    dataloader = DataLoader(dataset, batch_size=data_cfg['batch_size'],
                            shuffle=data_cfg['shuffle'], num_workers=data_cfg['num_workers'])

    # Load model
    model = Stage2EmotionModel(
        video_mae_name=stage2_cfg["video_mae_name"],
        num_faus=stage2_cfg["num_faus"],
        gnn_out_dim=stage2_cfg["gnn_out_dim"],
        num_emotions=stage2_cfg["num_emotions"],
        freeze_stage1=stage2_cfg["freeze_stage1"]
    ).to(device)

    # Load pretrained Stage 1 weights
    stage1_ckpt = config['stage1']['save_path']
    model.stage1_model.load_state_dict(torch.load(stage1_ckpt, map_location=device))
    print(f"✅ Loaded Stage 1 weights from {stage1_ckpt}")

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=stage2_cfg["learning_rate"],
                                  weight_decay=stage2_cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    # Train loop
    for epoch in range(stage2_cfg['num_epochs']):
        loss, acc, ccc = train_one_epoch(model, dataloader, optimizer, criterion, device)
        logger.log(f"Epoch {epoch+1}/{stage2_cfg['num_epochs']} | Loss: {loss:.4f} | Acc: {acc:.4f} | CCC: {ccc:.4f}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0 or (epoch + 1) == stage2_cfg['num_epochs']:
            torch.save(model.state_dict(), os.path.join(config['checkpoint_dir'], f"stage2_epoch{epoch+1}.pt"))

    # Final save
    torch.save(model.state_dict(), stage2_cfg['save_path'])
    logger.log(f"✅ Stage 2 complete. Final model saved to {stage2_cfg['save_path']}")


if __name__ == "__main__":
    train_stage2("config/config.yaml")
