import torch
import torch.nn as nn


class CrossAttentionModule(nn.Module):
    """
    Cross-Attention Module for refining VideoMAE features.
    Applies attention across temporal frames to capture inter-frame dependencies.
    """

    def __init__(self, input_dim=768, num_heads=8, dropout=0.1):
        """
        Args:
            input_dim (int): Dimensionality of VideoMAE features.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(CrossAttentionModule, self).__init__()

        self.norm = nn.LayerNorm(input_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.residual_fc = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input of shape (B, T, D) from VideoMAE (B=batch size, T=frames, D=feature dim)

        Returns:
            Tensor: Refined features of shape (B, T, D)
        """
        # Layer norm for stability
        x_norm = self.norm(x)

        # Self-attention across time (T dimension)
        attn_output, _ = self.cross_attn(x_norm, x_norm, x_norm)  # (B, T, D)

        # Add residual connection
        out = x + self.dropout(attn_output)
        out = out + self.residual_fc(out)  # Optional additional residual FC

        return out


# if __name__ == "__main__":
#     # Example usage
#     dummy_input = torch.randn(2, 16, 768)  # (B, T, D)
#     model = CrossAttentionModule()
#     with torch.no_grad():
#         output = model(dummy_input)
#     print("Output shape:", output.shape)  # (2, 16, 768)
