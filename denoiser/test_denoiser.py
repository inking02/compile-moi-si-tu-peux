"""
File: test_denoiser.py

Description: This script trains and evaluates the haze-removal denoiser by
building noisy NWPU image datasets, saving visual references and tensors, and
reporting reconstruction metrics.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.image import StructuralSimilarityIndexMeasure


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.NWPU.dataset_generator import create_anomaly_dataset
from classical_denoiser import Denoiser, FullDenoiser
from noisy_filter import add_haze


ORIGINAL_OUTPUT_DIR = REPO_ROOT / "denoiser" / "original_images"
NOISY_OUTPUT_DIR = REPO_ROOT / "denoiser" / "noisy_images"
DENOISED_OUTPUT_DIR = REPO_ROOT / "denoiser" / "denoised_images"
MODEL_OUTPUT_PATH = REPO_ROOT / "denoiser" / "denoiser_model.pt"

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
NUM_SAMPLES = 400


def save_image_references(
    images: torch.Tensor,
    anomaly_labels: torch.Tensor,
    photo_type_labels: torch.Tensor,
    split: str,
    output_dir: Path,
) -> None:
    """
    Saves image tensors as PNG references with labels in each filename.

    Args:
        images (torch.Tensor): Image batch shaped ``(N, C, H, W)``.
        anomaly_labels (torch.Tensor): Binary anomaly labels for each image.
        photo_type_labels (torch.Tensor): Photo-type labels for each image.
        split (str): Dataset split name used as the output subdirectory.
        output_dir (Path): Root directory where images should be saved.
    """
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    for idx, image in enumerate(images):
        image = image.detach().cpu().clamp(0, 1)
        array = (image.permute(1, 2, 0).numpy() * 255).astype("uint8")
        anomaly_label = int(anomaly_labels[idx])
        photo_type_label = int(photo_type_labels[idx])
        filename = f"{idx:03d}_anomaly-{anomaly_label}_type-{photo_type_label}.png"

        Image.fromarray(array).save(split_dir / filename)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes peak signal-to-noise ratio for normalized image tensors.

    Args:
        pred (torch.Tensor): Reconstructed image tensor.
        target (torch.Tensor): Clean target image tensor.

    Returns:
        float: PSNR value in decibels.
    """
    mse = F.mse_loss(pred, target)
    psnr = -10 * torch.log10(mse)
    return psnr.item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes structural similarity for normalized image tensors.

    Args:
        pred (torch.Tensor): Reconstructed image tensor.
        target (torch.Tensor): Clean target image tensor.

    Returns:
        float: SSIM score.
    """
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    score = ssim_metric(pred, target)
    return score.item()


def evaluate_reconstruction(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_fn: torch.nn.Module | None = None,
) -> tuple[float, float, float]:
    """
    Returns L1 loss, PSNR, and SSIM for reconstructed images.

    Args:
        pred (torch.Tensor): Reconstructed image tensor.
        target (torch.Tensor): Clean target image tensor.
        loss_fn (torch.nn.Module or None): Loss function used for reconstruction
            error, or ``None`` to use ``torch.nn.L1Loss``.

    Returns:
        tuple[float, float, float]: Reconstruction loss, PSNR, and SSIM.
    """
    if loss_fn is None:
        loss_fn = torch.nn.L1Loss()

    pred = pred.detach().cpu().clamp(0, 1)
    target = target.detach().cpu().clamp(0, 1)

    loss = loss_fn(pred, target).item()
    psnr = compute_psnr(pred, target)
    ssim = compute_ssim(pred, target)

    return loss, psnr, ssim


def evaluate_denoiser(
    model: Denoiser,
    noisy_images: torch.Tensor,
    clean_images: torch.Tensor,
    loss_fn: torch.nn.Module,
    batch_size: int = BATCH_SIZE,
    device: torch.device | None = None,
) -> tuple[float, float, float]:
    """
    Runs denoising in batches and returns reconstruction metrics.

    Args:
        model (Denoiser): Denoising model to evaluate.
        noisy_images (torch.Tensor): Noisy input image batch.
        clean_images (torch.Tensor): Clean target image batch.
        loss_fn (torch.nn.Module): Loss function used for reconstruction error.
        batch_size (int): Number of images per evaluation batch.
        device (torch.device or None): Device to use, or ``None`` to infer it.

    Returns:
        tuple[float, float, float]: Reconstruction loss, PSNR, and SSIM.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    denoised_batches = []

    with torch.no_grad():
        loader = DataLoader(
            TensorDataset(noisy_images, clean_images),
            batch_size=batch_size,
            shuffle=False,
        )

        for noisy_batch, _ in loader:
            denoised_batches.append(model(noisy_batch.to(device)).cpu())

    denoised_images = torch.cat(denoised_batches, dim=0)
    return evaluate_reconstruction(denoised_images, clean_images, loss_fn=loss_fn)


def train_denoiser(
    model: Denoiser,
    noisy_images: torch.Tensor,
    clean_images: torch.Tensor,
    val_noisy_images: torch.Tensor,
    val_clean_images: torch.Tensor,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> Denoiser:
    """
    Trains the denoiser and prints validation metrics after each epoch.

    Args:
        model (Denoiser): Denoising model to train.
        noisy_images (torch.Tensor): Noisy training image batch.
        clean_images (torch.Tensor): Clean training target image batch.
        val_noisy_images (torch.Tensor): Noisy validation image batch.
        val_clean_images (torch.Tensor): Clean validation target image batch.
        batch_size (int): Number of images per training batch.
        epochs (int): Number of training epochs.
        learning_rate (float): Adam optimizer learning rate.

    Returns:
        Denoiser: Trained model moved back to CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    train_loader = DataLoader(
        TensorDataset(noisy_images, clean_images),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.L1Loss()

    for epoch in range(epochs):
        total_loss = 0.0

        for noisy_batch, clean_batch in train_loader:
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)

            optimizer.zero_grad()
            denoised_batch = model(noisy_batch)
            loss = loss_fn(denoised_batch, clean_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * noisy_batch.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        val_loss, val_psnr, val_ssim = evaluate_denoiser(
            model,
            noisy_images=val_noisy_images,
            clean_images=val_clean_images,
            loss_fn=loss_fn,
            batch_size=batch_size,
            device=device,
        )

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"TrainLoss={avg_loss:.4f}, "
            f"ValLoss={val_loss:.4f}, "
            f"ValPSNR={val_psnr:.2f} dB, "
            f"ValSSIM={val_ssim:.3f}"
        )
        model.train()

    return model.cpu()


def save_dataset_outputs(
    noisy_train_dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    noisy_test_dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    denoised_train_dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    denoised_test_dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    model: Denoiser,
) -> None:
    """
    Persists noisy/denoised tensors and trained model weights.

    Args:
        noisy_train_dataset (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
            Noisy train images and labels.
        noisy_test_dataset (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
            Noisy test images and labels.
        denoised_train_dataset (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
            Denoised train images and labels.
        denoised_test_dataset (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
            Denoised test images and labels.
        model (Denoiser): Trained denoising model.
    """
    NOISY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DENOISED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(noisy_train_dataset, NOISY_OUTPUT_DIR / "noisy_train_dataset.pt")
    torch.save(noisy_test_dataset, NOISY_OUTPUT_DIR / "noisy_test_dataset.pt")
    torch.save(
        denoised_train_dataset,
        DENOISED_OUTPUT_DIR / "denoised_train_dataset.pt",
    )
    torch.save(
        denoised_test_dataset,
        DENOISED_OUTPUT_DIR / "denoised_test_dataset.pt",
    )
    torch.save(model.state_dict(), MODEL_OUTPUT_PATH)


def print_summary(
    clean_train_images: torch.Tensor,
    noisy_train_images: torch.Tensor,
    denoised_train_images: torch.Tensor,
    clean_test_images: torch.Tensor,
    noisy_test_images: torch.Tensor,
    denoised_test_images: torch.Tensor,
    results: dict[str, dict[str, float]],
) -> None:
    """
    Prints metric and tensor-shape summaries for the run.

    Args:
        clean_train_images (torch.Tensor): Clean training images.
        noisy_train_images (torch.Tensor): Noisy training images.
        denoised_train_images (torch.Tensor): Denoised training images.
        clean_test_images (torch.Tensor): Clean test images.
        noisy_test_images (torch.Tensor): Noisy test images.
        denoised_test_images (torch.Tensor): Denoised test images.
        results (dict[str, dict[str, float]]): Metrics grouped by split.
    """
    print("\n========== DENOISER SUMMARY ==========")

    for split, metrics in results.items():
        print(f"\n[{split.upper()}]")
        for metric_name, value in metrics.items():
            print(f"{metric_name:>12}: {value:.4f}")

    print(f"Clean train images: {tuple(clean_train_images.shape)}")
    print(f"Noisy train images: {tuple(noisy_train_images.shape)}")
    print(f"Denoised train images: {tuple(denoised_train_images.shape)}")
    print(f"Clean test images: {tuple(clean_test_images.shape)}")
    print(f"Noisy test images: {tuple(noisy_test_images.shape)}")
    print(f"Denoised test images: {tuple(denoised_test_images.shape)}")
    print(f"Noisy images saved to: {NOISY_OUTPUT_DIR}")
    print(f"Denoised images saved to: {DENOISED_OUTPUT_DIR}")
    print(f"Trained denoiser saved to: {MODEL_OUTPUT_PATH}")

    train = results["train"]
    test = results["test"]
    noisy_test = results["noisy_test"]
    improvement = results["improvement"]

    print(
        "Train: "
        f"Loss={train['loss']:.4f}, "
        f"PSNR={train['psnr']:.2f} dB, "
        f"SSIM={train['ssim']:.3f}"
    )
    print(
        "Test: "
        f"Loss={test['loss']:.4f}, "
        f"PSNR={test['psnr']:.2f} dB, "
        f"SSIM={test['ssim']:.3f}"
    )
    print(
        "Noisy -> Clean: "
        f"Loss={noisy_test['loss']:.4f}, "
        f"PSNR={noisy_test['psnr']:.2f} dB, "
        f"SSIM={noisy_test['ssim']:.3f}"
    )
    print(
        "Denoised -> Clean: "
        f"Loss={test['loss']:.4f}, "
        f"PSNR={test['psnr']:.2f} dB, "
        f"SSIM={test['ssim']:.3f}"
    )
    print(
        f"Improvement: +{improvement['psnr_gain']:.2f} dB PSNR, "
        f"+{improvement['ssim_gain']:.3f} SSIM"
    )


def main() -> None:
    """Runs the full denoiser training, export, and evaluation workflow."""
    clean_train_dataset, clean_test_dataset = create_anomaly_dataset(
        num_samples=NUM_SAMPLES,
        normalize=False,
    )

    clean_train_images, train_anomaly_labels, train_photo_type_labels = (
        clean_train_dataset
    )
    clean_test_images, test_anomaly_labels, test_photo_type_labels = clean_test_dataset

    noisy_train_images = torch.stack([add_haze(image) for image in clean_train_images])
    noisy_test_images = torch.stack([add_haze(image) for image in clean_test_images])

    save_image_references(
        clean_train_images,
        train_anomaly_labels,
        train_photo_type_labels,
        "train",
        output_dir=ORIGINAL_OUTPUT_DIR,
    )
    save_image_references(
        clean_test_images,
        test_anomaly_labels,
        test_photo_type_labels,
        "test",
        output_dir=ORIGINAL_OUTPUT_DIR,
    )
    save_image_references(
        noisy_train_images,
        train_anomaly_labels,
        train_photo_type_labels,
        "train",
        output_dir=NOISY_OUTPUT_DIR,
    )
    save_image_references(
        noisy_test_images,
        test_anomaly_labels,
        test_photo_type_labels,
        "test",
        output_dir=NOISY_OUTPUT_DIR,
    )

    denoiser = train_denoiser(
        FullDenoiser(),
        noisy_images=noisy_train_images,
        clean_images=clean_train_images,
        val_noisy_images=noisy_test_images,
        val_clean_images=clean_test_images,
    )
    denoiser.eval()

    with torch.no_grad():
        denoised_train_images = denoiser(noisy_train_images)
        denoised_test_images = denoiser(noisy_test_images)

    save_image_references(
        denoised_train_images,
        train_anomaly_labels,
        train_photo_type_labels,
        "train",
        output_dir=DENOISED_OUTPUT_DIR,
    )
    save_image_references(
        denoised_test_images,
        test_anomaly_labels,
        test_photo_type_labels,
        "test",
        output_dir=DENOISED_OUTPUT_DIR,
    )

    noisy_train_dataset = (
        noisy_train_images,
        train_anomaly_labels,
        train_photo_type_labels,
    )
    noisy_test_dataset = (
        noisy_test_images,
        test_anomaly_labels,
        test_photo_type_labels,
    )
    denoised_train_dataset = (
        denoised_train_images,
        train_anomaly_labels,
        train_photo_type_labels,
    )
    denoised_test_dataset = (
        denoised_test_images,
        test_anomaly_labels,
        test_photo_type_labels,
    )
    save_dataset_outputs(
        noisy_train_dataset,
        noisy_test_dataset,
        denoised_train_dataset,
        denoised_test_dataset,
        denoiser,
    )

    train_loss, train_psnr, train_ssim = evaluate_reconstruction(
        denoised_train_images,
        clean_train_images,
    )
    test_loss, test_psnr, test_ssim = evaluate_reconstruction(
        denoised_test_images,
        clean_test_images,
    )
    noisy_test_loss, noisy_test_psnr, noisy_test_ssim = evaluate_reconstruction(
        noisy_test_images,
        clean_test_images,
    )

    results = {
        "train": {
            "loss": train_loss,
            "psnr": train_psnr,
            "ssim": train_ssim,
        },
        "test": {
            "loss": test_loss,
            "psnr": test_psnr,
            "ssim": test_ssim,
        },
        "noisy_test": {
            "loss": noisy_test_loss,
            "psnr": noisy_test_psnr,
            "ssim": noisy_test_ssim,
        },
        "improvement": {
            "psnr_gain": test_psnr - noisy_test_psnr,
            "ssim_gain": test_ssim - noisy_test_ssim,
        },
    }

    print_summary(
        clean_train_images,
        noisy_train_images,
        denoised_train_images,
        clean_test_images,
        noisy_test_images,
        denoised_test_images,
        results,
    )


if __name__ == "__main__":
    main()
