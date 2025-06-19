import torch
import torch.nn as nn

from model_stage1 import Stage1FAUModel
from models.emotion_classifier import EmotionClassifier


class Stage2EmotionModel(nn.Module):
    """
    Stage 2: Emotion classification model that uses:
    - FAU detection output (logits)
    - GNN co-occurrence features
    from Stage 1 to predict emotions.
    """

    def __init__(self,
                 video_mae_name="MCG-NJU/videomae-base",
                 num_faus=12,
                 gnn_out_dim=256,
                 num_emotions=7,
                 freeze_stage1=True):
        """
        Args:
            video_mae_name (str): Path/name of pre-trained VideoMAE
            num_faus (int): Number of FAUs (output of FAU head)
            gnn_out_dim (int): Dim of GNN output
            num_emotions (int): Number of target emotion classes
            freeze_stage1 (bool): Whether to freeze Stage 1 weights
        """
        super(Stage2EmotionModel, self).__init__()

        self.stage1_model = Stage1FAUModel(
            video_mae_name=video_mae_name,
            num_faus=num_faus,
            gnn_out_dim=gnn_out_dim,
            freeze_video_mae=True  # Still frozen, consistent with Stage 1
        )

        if freeze_stage1:
            for param in self.stage1_model.parameters():
                param.requires_grad = False

        self.emotion_head = EmotionClassifier(
            fau_feat_dim=num_faus,
            gnn_feat_dim=gnn_out_dim,
            num_emotions=num_emotions
        )

    def forward(self, video_clip, edge_index):
        """
        Args:
            video_clip (Tensor): (B, T, C, H, W)
            edge_index (LongTensor): Edge list for GNN

        Returns:
            Tensor: Emotion logits (B, T, num_emotions)
        """
        fau_logits, gnn_out = self.stage1_model(video_clip, edge_index)  # Both are (B, T, *)
        emotion_logits = self.emotion_head(fau_logits, gnn_out)          # (B, T, num_emotions)
        return emotion_logits


if __name__ == "__main__":
    # Test run
    dummy_video = torch.randn(2, 16, 3, 224, 224)
    edge_index = torch.tensor([[i, i+1] for i in range(31)] +
                              [[i+1, i] for i in range(31)], dtype=torch.long).t()

    model = Stage2EmotionModel(num_faus=12, gnn_out_dim=256, num_emotions=7)
    with torch.no_grad():
        emotion_out = model(dummy_video, edge_index)
    print("Emotion logits shape:", emotion_out.shape)  # (2, 16, 7)
