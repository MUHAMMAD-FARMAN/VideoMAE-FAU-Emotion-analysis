import torch
import torch.nn as nn
from transformers import VideoMAEModel, VideoMAEImageProcessor


class VideoMAEWrapper(nn.Module):
    """
    A wrapper around a pre-trained VideoMAE model for feature extraction.
    """

    def __init__(self, model_name: str = "MCG-NJU/videomae-base", output_dim: int = 768, freeze: bool = True):
        """
        Args:
            model_name (str): Name or path to the pretrained VideoMAE model.
            output_dim (int): Dimensionality of output features.
            freeze (bool): Whether to freeze the VideoMAE backbone during training.
        """
        super(VideoMAEWrapper, self).__init__()
        self.model_name = model_name
        self.output_dim = output_dim

        # Load pre-trained VideoMAE backbone
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.backbone = VideoMAEModel.from_pretrained(model_name)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract features from video.

        Args:
            video_tensor (Tensor): Shape (B, T, C, H, W) — batch of video clips

        Returns:
            Tensor: Features of shape (B, T, D)
        """
        B, T, C, H, W = video_tensor.shape
        video_tensor = video_tensor.view(-1, C, H, W)  # (B*T, C, H, W)

        # Preprocess using processor (if needed)
        inputs = self.processor(list(video_tensor), return_tensors="pt")
        inputs = {k: v.to(video_tensor.device) for k, v in inputs.items()}

        # Extract features
        outputs = self.backbone(**inputs)
        last_hidden = outputs.last_hidden_state  # (B*T, num_patches+1, D)

        # Use [CLS] token for each frame
        cls_tokens = last_hidden[:, 0, :]  # (B*T, D)
        cls_tokens = cls_tokens.view(B, T, -1)  # (B, T, D)

        return cls_tokens

    def get_processor(self):
        return self.processor


# if __name__ == "__main__":
#     # Example usage
#     model = VideoMAEWrapper()
#     dummy_input = torch.randn(2, 16, 3, 224, 224)  # (B, T, C, H, W)
#     with torch.no_grad():
#         features = model(dummy_input)
#     print("Extracted feature shape:", features.shape)  # (2, 16, 768)
