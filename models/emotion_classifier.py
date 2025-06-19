import torch
import torch.nn as nn


class EmotionClassifier(nn.Module):
    """
    Classifies emotions using features from:
    - FAU detection head (multi-label outputs or internal features)
    - FAU co-occurrence GNN output
    """

    def __init__(self,
                 fau_feat_dim=12,        # Number of FAUs (or FC output dim)
                 gnn_feat_dim=256,       # Output dim from GNN head
                 hidden_dim=128,
                 num_emotions=7,
                 dropout=0.3):
        """
        Args:
            fau_feat_dim (int): Input dimension from FAU detection branch.
            gnn_feat_dim (int): Input dimension from GNN branch.
            hidden_dim (int): Hidden FC layer size.
            num_emotions (int): Number of emotion classes.
            dropout (float): Dropout probability.
        """
        super(EmotionClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(fau_feat_dim + gnn_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_emotions)
        )

    def forward(self, fau_features, gnn_features):
        """
        Args:
            fau_features (Tensor): (B, T, fau_feat_dim)
            gnn_features (Tensor): (B, T, gnn_feat_dim)

        Returns:
            Tensor: Emotion logits (B, T, num_emotions)
        """
        x = torch.cat([fau_features, gnn_features], dim=-1)  # (B, T, fau+gnn)
        return self.classifier(x)


if __name__ == "__main__":
    # Test run
    fau_feats = torch.randn(2, 16, 12)     # (B, T, 12)
    gnn_feats = torch.randn(2, 16, 256)    # (B, T, 256)
    model = EmotionClassifier(fau_feat_dim=12, gnn_feat_dim=256, num_emotions=7)

    with torch.no_grad():
        out = model(fau_feats, gnn_feats)
    print("Emotion logits shape:", out.shape)  # (2, 16, 7)
