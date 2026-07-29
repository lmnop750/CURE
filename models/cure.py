import copy

import torch
import torch.nn as nn

from .if_decoder import IFDecoder


class CURE(nn.Module):
    def __init__(self, feature_dim=768, embedding_dim=64):
        super().__init__()
        self.student = IFDecoder(feature_dim, embedding_dim)
        self.teacher = copy.deepcopy(self.student)
        self.teacher.requires_grad_(False)

    def forward_student(self, features, guidance_map):
        return self.student(features, guidance_map=guidance_map)

    @torch.no_grad()
    def forward_teacher(self, features):
        foreground_logits, _ = self.teacher(features)
        return foreground_logits

    def forward(self, features, teacher=False, guidance_map=None):
        if teacher:
            return self.forward_teacher(features)
        return self.forward_student(features, guidance_map)
