import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from torchmetrics.classification import BinaryJaccardIndex


class SegmentationModel(pl.LightningModule):
    """PyTorch Lightning module for semantic segmentation using U-Net.

    Model wraps a U-Net with pretrained encoder backbone (default: ResNet34).
    Uses combination of Binary Cross-Entropy (BCE) and Dice loss to
    achieve pixel-level accuracy and region-level consistency.

    Attributes:
        model (nn.Module): U-Net architecture from segmentation_models_pytorch.
        std (torch.Tensor): Per-channel standard deviation  for input normalisation.
        mean (torch.Tensor): Per-channel mean for input normalisation.
        lr (float): Optimiser learning rate.
        bce_weight (float): Weight factor for BCE loss in loss calculation
    """

    def __init__(
        self, encoder_name: str = "resnet34", lr: float = 1e-3, bce_weight: float = 0.5
    ):
        """Initialises the segmentation model with a given encoder.

        Args:
            encoder_name (str, optional): The backbone architecture for the U-Net encoder.
                Defaults to "resnet34". Examples: "resnet50", "efficientnet-b0".
            lr (float, optional): Initial learning rate for the optimiser. Defaults to 1e-3.
            bce_weight (float, optional): Weight for BCE loss in the final loss calculation.
                Dice loss weight is computed as (1 - bce_weight). Defaults to 0.5.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            in_channels=3,
            classes=1,
            encoder_weights="imagenet",
        )

        # Register normalisation buffers for consistent preprocessing
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # Freeze encoder to prevent overwriting pretrained weights at the start
        for layer in self.model.encoder.parameters():
            layer.requires_grad = False

        self.lr = lr
        self.bce_weight = bce_weight

        # Loss functions
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        # Metrics
        self.train_iou = BinaryJaccardIndex()
        self.val_iou = BinaryJaccardIndex()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): Input tensor with format (B, C, H, W)

        Returns:
            torch.Tensor: Raw logits of shape (B, 1, H, W)
        """
        x = (x - self.mean) / self.std
        return self.model(x)

    def compute_loss(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Combined BCE + Dice loss.

        Args:
            logits (torch.Tensor): Raw model logits of shape (B, 1, H, W).
            masks (torch.Tensor): Ground truth of shape (B, 1, H, W).

        Returns:
            torch.Tensor
        """
        bce_loss = self.bce(logits, masks)
        dice_loss = self.dice(logits, masks)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """

        Args:
            batch (tuple): A tuple `(images, masks)` where:
                - images: Tensor of input images (B, C, H, W).
                - masks: Tensor of ground truth masks (B, 1, H, W).
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor
        """
        images, masks = batch
        logits = self(images)

        loss = self.compute_loss(logits, masks)
        iou = self.train_iou(logits, masks)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_iou", iou, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx: int):
        """

        Args:
            batch (tuple): A tuple `(images, masks)` similar to training_step.
            batch_idx (int): Index of the current batch.
        """
        images, masks = batch
        logits = self(images)

        loss = self.compute_loss(logits, masks)
        iou = self.val_iou(logits, masks)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_iou", iou, prog_bar=True)

    def configure_optimizers(self):
        """Configures optimiser and learning rate scheduler.

        Adam optimiser + cosine annealing schedule.
        Helps explore more aggressively at the start and fine-tune smoothly toward the end.

        Returns:
            tuple: A tuple containing:
                - optimiser: Configured optimiser instance.
                - scheduler: Configured learning rate scheduler.
        """
        optimiser = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=10, eta_min=1e-5
        )

        return [optimiser], [scheduler]
