import torch
import torch.nn as nn
import torch.nn.functional as F


class IFDecoder(nn.Module):
    def __init__(self, feature_dim=768, embedding_dim=64):
        super().__init__()
        self.base_proj = nn.Conv2d(feature_dim, embedding_dim, kernel_size=1)
        self.coarse_head = nn.Conv2d(embedding_dim, 1, kernel_size=1)
        self.feedback_processor = nn.Sequential(
            nn.Conv2d(embedding_dim + 1, embedding_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                embedding_dim,
                embedding_dim,
                kernel_size=3,
                padding=2,
                dilation=2,
            ),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(embedding_dim, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.foreground_head = nn.Conv2d(embedding_dim, 1, kernel_size=1)
        self.background_head = nn.Conv2d(embedding_dim, 1, kernel_size=1)

    def forward(self, features, guidance_map=None):
        if isinstance(features, (list, tuple)):
            features = features[-1]

        base_feature = self.base_proj(features)
        batch_size, _, height, width = base_feature.shape

        if guidance_map is None:
            coarse_probability = torch.sigmoid(self.coarse_head(base_feature))
        else:
            coarse_probability = F.interpolate(
                guidance_map,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        confidence = (2.0 * torch.abs(coarse_probability - 0.5)).square()
        uncertainty = 1.0 - confidence
        maximum = uncertainty.flatten(1).amax(dim=1).view(batch_size, 1, 1, 1)
        normalized_uncertainty = uncertainty / (maximum + 1e-5)

        feedback_input = torch.cat(
            [base_feature, normalized_uncertainty],
            dim=1,
        )
        refined_feature = self.feedback_processor(feedback_input)
        attention = self.spatial_attention(refined_feature)
        fused_feature = base_feature + refined_feature * attention

        foreground_logits = self.foreground_head(fused_feature)
        background_logits = self.background_head(fused_feature)
        return foreground_logits, background_logits
