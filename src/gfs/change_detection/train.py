"""One-shot Siamese training loops for the change-detection models (paper §3.4).

The paper trains each Siamese model for a single epoch (``CD_EPOCHS``) — except
the customized Res-Net, trained for ``CD_EPOCHS_RESNET`` — with Adam at
``CD_LEARNING_RATE`` and an MSE reconstruction objective, in batches of
``CD_BATCH_SIZE`` over 256x256 imagelets. There are no change labels: the network
learns to reconstruct one date, and the difference of its outputs/features
between the two dates is the change signal ("one-shot Siamese learning").

Two training shapes are provided, mirroring the notebook:

- :func:`train_siamese` — patch-based loop for models with a ``forward(x1, x2)``
  signature (TinyCD, CGNet, BiDateNet, FC-SiamDiff).
- :func:`train_resnet_band` — per-band autoencoder loop for the Res-Net
  ``FeatureExtractor`` (``forward(x)``), trained for many epochs.
"""

from __future__ import annotations

import torch
import torch.optim as optim
from loguru import logger
from torch import Tensor, nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

from gfs.change_detection.common import (
    ClearCache,
    extract_patches,
    resize_to_match,
    select_device,
)
from gfs.change_detection.models.resnet import FeatureExtractor
from gfs.config import (
    CD_BATCH_SIZE,
    CD_EPOCHS,
    CD_EPOCHS_RESNET,
    CD_LEARNING_RATE,
    PATCH_SIZE,
)


def _linear_decay_scheduler(optimizer: optim.Optimizer, max_epochs: int) -> LambdaLR:
    """The notebook's ``1 - epoch/(max_epochs+1)`` linear lr decay."""
    return LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 - epoch / float(max_epochs + 1))


def _stack_patches(image_tensor: Tensor, patch_size: int, stride: int) -> Tensor:
    """Extract patches from a ``(1, C, H, W)`` tensor and stack to ``(N, C, h, w)``."""
    patches = extract_patches(image_tensor.squeeze(0), patch_size, stride)
    stacked = torch.stack([p.clone().detach().unsqueeze(0) for p in patches])
    return stacked.squeeze(1)


def train_siamese(
    model: nn.Module,
    im1_tensor: Tensor,
    im2_tensor: Tensor,
    *,
    save_path: str | None = None,
    patch_size: int = PATCH_SIZE,
    stride: int | None = None,
    batch_size: int = CD_BATCH_SIZE,
    n_epochs: int = CD_EPOCHS,
    lr: float = CD_LEARNING_RATE,
    device: torch.device | None = None,
) -> nn.Module:
    """One-shot Siamese training (Adam, MSE) over 256x256 imagelets.

    The model reconstructs ``im1`` from the ``(im1, im2)`` pair; the MSE between
    its resized output and the ``im1`` patches is minimized. Faithful to the
    notebook's TinyCD/CGNet/BiDateNet training cells. Optionally saves the
    trained ``state_dict`` to ``save_path``.
    """
    if device is None:
        device = select_device()
    if stride is None:
        stride = patch_size // 2
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 - epoch / float(50 + 1))

    im1_patches = _stack_patches(im1_tensor, patch_size, stride).to(device)
    im2_patches = _stack_patches(im2_tensor, patch_size, stride).to(device)

    dataset = TensorDataset(im1_patches, im2_patches)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        loss = torch.tensor(0.0)
        with ClearCache():
            for im1_batch, im2_batch in dataloader:
                im1_batch = im1_batch.to(device)
                im2_batch = im2_batch.to(device)
                output = model(im1_batch, im2_batch)
                resized_output = resize_to_match(output, im1_batch)
                loss = criterion(resized_output, im1_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        scheduler.step()
        logger.info(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")
    return model


def train_resnet_band(
    im1_tensor: Tensor,
    im2_tensor: Tensor,
    *,
    ngf: int = 1,
    n_blocks: int = 4,
    n_epochs: int = CD_EPOCHS_RESNET,
    lr: float = CD_LEARNING_RATE,
    save_path: str | None = None,
    device: torch.device | None = None,
) -> FeatureExtractor:
    """Per-band Res-Net autoencoder training (Adam, MSE, many epochs).

    The :class:`FeatureExtractor` is trained to map the (single-band) ``im1``
    composite onto ``im2`` (reconstruction target), as in the notebook's Res-Net
    ``train_and_save_features``. ``im1_tensor`` / ``im2_tensor`` are ``(1, 1, H, W)``.
    """
    if device is None:
        device = select_device()
    model = FeatureExtractor(
        input_nc=1, output_nc=1, ngf=ngf, use_dropout=False, n_blocks=n_blocks
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = _linear_decay_scheduler(optimizer, n_epochs)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        outputs = model(im1_tensor)
        target_resized = resize_to_match(im2_tensor, outputs)
        loss = criterion(outputs, target_resized)
        loss.backward()
        optimizer.step()
        scheduler.step()
        logger.info(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")
    return model
