import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class FAUCooccurrenceGNN(nn.Module):
    """
    Graph-based model to learn FAU co-occurrence patterns.
    Takes cross-attention features as input and passes them through a GCN.
    """

    def __init__(self, input_dim=768, hidden_dim=512, output_dim=256, num_layers=2, dropout=0.3):
        """
        Args:
            input_dim (int): Dim of input features from CrossAttentionModule.
            hidden_dim (int): Hidden layer dim of GNN.
            output_dim (int): Final output representation dim.
            num_layers (int): Number of GCN layers.
            dropout (float): Dropout rate.
        """
        super(FAUCooccurrenceGNN, self).__init__()

        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

        self.gcn_layers.append(GCNConv(hidden_dim, output_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        """
        Args:
            x (Tensor): Node features of shape (B*T, D)
            edge_index (LongTensor): Graph edge indices (2, num_edges)

        Returns:
            Tensor: Output node features (B*T, output_dim)
        """
        for i, layer in enumerate(self.gcn_layers):
            x = layer(x, edge_index)
            if i != len(self.gcn_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x


if __name__ == "__main__":
    # Example usage
    from torch_geometric.data import Data

    B, T, D = 2, 16, 768
    total_nodes = B * T

    dummy_x = torch.randn(total_nodes, D)
    # Simple edge_index (connect each node to next for demo)
    edge_index = torch.tensor([[i, i+1] for i in range(total_nodes - 1)] +
                              [[i+1, i] for i in range(total_nodes - 1)], dtype=torch.long).t()

    model = FAUCooccurrenceGNN(input_dim=768, output_dim=256)
    with torch.no_grad():
        out = model(dummy_x, edge_index)
    print("Output shape:", out.shape)  # (B*T, 256)
