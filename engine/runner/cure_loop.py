from typing import Dict, Optional

import torch
import torch.nn.functional as F

from configs import CUREConfig
from models import CURE


class CURETrainLoop:
    def __init__(
        self,
        model: CURE,
        optimizer: torch.optim.Optimizer,
        config: Optional[CUREConfig] = None,
        scheduler=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config or CUREConfig()
        self.scheduler = scheduler
        self.epoch = 0
        self.global_step = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _spatial_level_weighting(
        self,
        initial_masks,
        teacher_logits,
        clean_logits,
        perturbed_logits,
    ):
        initial_target = (initial_masks > 0.5).float()
        if self.epoch < self.config.warmup_epochs:
            return initial_target, torch.ones_like(initial_target), 0.0, 0.0

        teacher_probability = torch.sigmoid(teacher_logits.detach())
        clean_probability = torch.sigmoid(clean_logits.detach())
        perturbed_probability = torch.sigmoid(perturbed_logits.detach())

        discrepancy = torch.abs(clean_probability - perturbed_probability)
        teacher_confidence = 2.0 * torch.abs(teacher_probability - 0.5)
        spatial_weight = (1.0 - discrepancy).square() * teacher_confidence
        target = (teacher_probability > 0.5).float()

        roi = (teacher_probability > self.config.roi_threshold).float()
        if roi.sum() > 0:
            teacher_state = (teacher_confidence * roi).sum() / roi.sum()
            student_state = ((1.0 - discrepancy) * roi).sum() / roi.sum()
        else:
            teacher_state = teacher_confidence.mean()
            student_state = (1.0 - discrepancy).mean()

        return (
            target,
            spatial_weight,
            teacher_state.item(),
            student_state.item(),
        )

    def _weighted_bce(self, logits, targets, weights):
        element_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        return (element_loss * weights).sum() / (
            weights.sum() + self.config.eps
        )

    @torch.no_grad()
    def _update_teacher(self, teacher_state, student_state):
        base_momentum = min(
            1.0 - 1.0 / (self.global_step + 1.0),
            self.config.ema_weight,
        )

        if teacher_state > 0.0:
            state_ratio = student_state / (teacher_state + self.config.eps)
            scale = state_ratio ** self.config.scale_power
            scale = min(max(scale, self.config.clip_min), self.config.clip_max)
            momentum = 1.0 - (1.0 - base_momentum) * scale
            momentum = min(max(momentum, 0.0), 1.0)
        else:
            momentum = base_momentum

        for teacher_parameter, student_parameter in zip(
            self.model.teacher.parameters(),
            self.model.student.parameters(),
        ):
            teacher_parameter.mul_(momentum).add_(
                student_parameter,
                alpha=1.0 - momentum,
            )

        for teacher_buffer, student_buffer in zip(
            self.model.teacher.buffers(),
            self.model.student.buffers(),
        ):
            teacher_buffer.copy_(student_buffer)

        return momentum

    def train_step(
        self,
        features: torch.Tensor,
        initial_masks: torch.Tensor,
    ) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        size = (self.config.feature_size, self.config.feature_size)
        features = F.interpolate(
            features,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        initial_masks = F.interpolate(
            initial_masks.float(),
            size=size,
            mode="bilinear",
            align_corners=False,
        )

        with torch.no_grad():
            teacher_logits = self.model.forward_teacher(features)
            teacher_probability = torch.sigmoid(teacher_logits)

        perturbed_features = F.dropout2d(
            features,
            p=self.config.dropout_rate,
            training=True,
        )
        paired_features = torch.cat([features, perturbed_features], dim=0)
        paired_guidance = torch.cat(
            [teacher_probability, teacher_probability],
            dim=0,
        )

        foreground_logits, background_logits = self.model.forward_student(
            paired_features,
            paired_guidance,
        )
        clean_logits, perturbed_logits = foreground_logits.chunk(2, dim=0)
        clean_background, perturbed_background = background_logits.chunk(
            2,
            dim=0,
        )

        target, spatial_weight, teacher_state, student_state = (
            self._spatial_level_weighting(
                initial_masks,
                teacher_logits,
                clean_logits,
                perturbed_logits,
            )
        )

        loss_main = self._weighted_bce(
            clean_logits,
            target,
            spatial_weight,
        ) + self._weighted_bce(
            clean_background,
            1.0 - target,
            spatial_weight,
        )
        loss_perturbation = self._weighted_bce(
            perturbed_logits,
            target,
            spatial_weight,
        ) + self._weighted_bce(
            perturbed_background,
            1.0 - target,
            spatial_weight,
        )

        teacher_confidence = 2.0 * torch.abs(teacher_probability - 0.5)
        teacher_uncertainty = 1.0 - teacher_confidence
        clean_probability = torch.sigmoid(clean_logits)
        entropy = -(
            clean_probability * torch.log(clean_probability + self.config.eps)
            + (1.0 - clean_probability)
            * torch.log(1.0 - clean_probability + self.config.eps)
        )
        loss_entropy = (entropy * teacher_uncertainty).mean()

        loss = (
            loss_main
            + self.config.lambda_fp * loss_perturbation
            + self.config.lambda_ent * loss_entropy
        )
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        momentum = self._update_teacher(teacher_state, student_state)
        self.global_step += 1

        return {
            "loss": loss.item(),
            "loss_main": loss_main.item(),
            "loss_perturbation": loss_perturbation.item(),
            "loss_entropy": loss_entropy.item(),
            "mean_spatial_weight": spatial_weight.mean().item(),
            "teacher_momentum": momentum,
        }
