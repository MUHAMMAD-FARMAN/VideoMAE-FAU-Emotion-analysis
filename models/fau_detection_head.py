import torch
import torch.nn as nn


class FAUDetectionHead(nn.Module):
    """
    Fully Connected (FC) based head for predicting FAUs from refined video features.
    """

    def __init__(self, input_dim=768, hidden_dim=512, num_faus=12, dropout=0.3):
        """
        Args:
            input_dim (int): Dimensionality of input features (from CrossAttentionModule).
            hidden_dim (int): Hidden layer size.
            num_faus (int): Number of Facial Action Units to predict.
            dropout (float): Dropout probability.
        """
        super(FAUDetectionHead, self).__init__()

        self.fau_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_faus),
            nn.Sigmoid()  # Multi-label prediction: output ∈ [0,1] for each FAU
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Refined features of shape (B, T, D)

        Returns:
            Tensor: FAU predictions of shape (B, T, num_faus)
        """
        return self.fau_classifier(x)


# if __name__ == "__main__":
#     # Example usage
#     dummy_input = torch.randn(2, 16, 768)  # (B, T, D)
#     model = FAUDetectionHead(input_dim=768, num_faus=12)
#     with torch.no_grad():
#         output = model(dummy_input)
#     print("FAU Predictions shape:", output.shape)  # (2, 16, 12)
