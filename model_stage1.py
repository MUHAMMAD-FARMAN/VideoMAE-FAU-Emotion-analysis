import torch
import torch.nn as nn

from models.video_mae import VideoMAEWrapper
from models.cross_attention import CrossAttentionModule
from models.fau_detection_head import FAUDetectionHead
from models.fau_cooccur_gnn import FAUCooccurrenceGNN


class Stage1FAUModel(nn.Module):
    """
    Stage 1: End-to-end model for FAU detection and co-occurrence modeling.
    """

    def __init__(self,
                 video_mae_name="MCG-NJU/videomae-base",
                 num_faus=12,
                 feature_dim=768,
                 gnn_out_dim=256,
                 freeze_video_mae=True):
        """
        Args:
            video_mae_name (str): Name/path of pre-trained VideoMAE.
            num_faus (int): Number of FAUs to detect.
            feature_dim (int): Output dim of VideoMAE.
            gnn_out_dim (int): Output dim of co-occurrence GNN.
            freeze_video_mae (bool): Whether to freeze VideoMAE.
        """
        super(Stage1FAUModel, self).__init__()

        self.video_encoder = VideoMAEWrapper(model_name=video_mae_name,
                                             output_dim=feature_dim,
                                             freeze=freeze_video_mae)

        self.cross_attention = CrossAttentionModule(input_dim=feature_dim)
        self.fau_head = FAUDetectionHead(input_dim=feature_dim, num_faus=num_faus)
        self.gnn_head = FAUCooccurrenceGNN(input_dim=feature_dim, output_dim=gnn_out_dim)

    def forward(self, video_clip, edge_index):
        """
        Args:
            video_clip (Tensor): (B, T, C, H, W) input video
            edge_index (LongTensor): Graph edges for GNN (2, num_edges)

        Returns:
            Tuple:
                - fau_logits: (B, T, num_faus)
                - gnn_features: (B, T, gnn_out_dim)
        """
        B, T, C, H, W = video_clip.shape

        # Stage 1: VideoMAE → Cross Attention
        x = self.video_encoder(video_clip)         # (B, T, D)
        x = self.cross_attention(x)                # (B, T, D)

        # FAU Detection
        fau_logits = self.fau_head(x)              # (B, T, num_faus)

        # GNN Co-occurrence Modeling
        gnn_input = x.view(B * T, -1)              # (B*T, D)
        gnn_out = self.gnn_head(gnn_input, edge_index)  # (B*T, gnn_out_dim)
        gnn_out = gnn_out.view(B, T, -1)           # (B, T, gnn_out_dim)

        return fau_logits, gnn_out


if __name__ == "__main__":
    # Test run
    dummy_video = torch.randn(2, 16, 3, 224, 224)  # (B, T, C, H, W)
    edge_index = torch.tensor([[i, i+1] for i in range(31)] +
                              [[i+1, i] for i in range(31)], dtype=torch.long).t()

    model = Stage1FAUModel(num_faus=12)
    with torch.no_grad():
        fau_pred, gnn_out = model(dummy_video, edge_index)
    print("FAU shape:", fau_pred.shape)     # (2, 16, 12)
    print("GNN output shape:", gnn_out.shape)  # (2, 16, 256)
