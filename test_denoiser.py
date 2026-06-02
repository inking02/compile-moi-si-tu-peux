from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure

from data.NWPU.dataset_generator import create_anomaly_dataset

from denoiser.noisy_filter import add_haze
from denoiser.classical_denoiser import Denoiser

NOISY_OUTPUT_DIR = Path("denoiser/noisy_images")
DENOISED_OUTPUT_DIR = Path("denoiser/denoised_images")
MODEL_OUTPUT_PATH = Path("denoiser/denoiser_model.pt")
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3


def save_image_references(
    images: torch.Tensor,
    anomaly_labels: torch.Tensor,
    photo_type_labels: torch.Tensor,
    split: str,
    output_dir: Path = NOISY_OUTPUT_DIR,
) -> None:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    for idx, image in enumerate(images):
        image = image.detach().cpu().clamp(0, 1)
        array = (image.permute(1, 2, 0).numpy() * 255).astype("uint8")
        anomaly_label = int(anomaly_labels[idx])
        photo_type_label = int(photo_type_labels[idx])
        filename = f"{idx:03d}_anomaly-{anomaly_label}_type-{photo_type_label}.png"
        Image.fromarray(array).save(split_dir / filename)


def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    psnr = -10 * torch.log10(mse)
    return psnr.item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    score = ssim_metric(pred, target)
    return score.item()


def evaluate_reconstruction(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_fn: torch.nn.Module | None = None,
) -> tuple[float, float, float]:
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
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    denoised_batches = []
    with torch.no_grad():
        for noisy_batch, _ in DataLoader(
            TensorDataset(noisy_images, clean_images),
            batch_size=batch_size,
            shuffle=False,
        ):
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


clean_train_dataset, clean_test_dataset = create_anomaly_dataset(normalize=False)

clean_train_images, train_anomaly_labels, train_photo_type_labels = clean_train_dataset
clean_test_images, test_anomaly_labels, test_photo_type_labels = clean_test_dataset

noisy_train_images = torch.stack([add_haze(image) for image in clean_train_images])
noisy_test_images = torch.stack([add_haze(image) for image in clean_test_images])

save_image_references(
    noisy_train_images, train_anomaly_labels, train_photo_type_labels, "train"
)
save_image_references(
    noisy_test_images, test_anomaly_labels, test_photo_type_labels, "test"
)

denoiser = Denoiser(image_size=32, RGB=True)
denoiser = train_denoiser(
    denoiser,
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
noisy_test_dataset = (noisy_test_images, test_anomaly_labels, test_photo_type_labels)
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

NOISY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DENOISED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
torch.save(noisy_train_dataset, NOISY_OUTPUT_DIR / "noisy_train_dataset.pt")
torch.save(noisy_test_dataset, NOISY_OUTPUT_DIR / "noisy_test_dataset.pt")
torch.save(denoised_train_dataset, DENOISED_OUTPUT_DIR / "denoised_train_dataset.pt")
torch.save(denoised_test_dataset, DENOISED_OUTPUT_DIR / "denoised_test_dataset.pt")
torch.save(denoiser.state_dict(), MODEL_OUTPUT_PATH)

train_loss, train_psnr, train_ssim = evaluate_reconstruction(
    denoised_train_images, clean_train_images
)
test_loss, test_psnr, test_ssim = evaluate_reconstruction(
    denoised_test_images, clean_test_images
)
noisy_test_loss, noisy_test_psnr, noisy_test_ssim = evaluate_reconstruction(
    noisy_test_images, clean_test_images
)
psnr_improvement = test_psnr - noisy_test_psnr
ssim_improvement = test_ssim - noisy_test_ssim

print(f"Clean train images: {tuple(clean_train_images.shape)}")
print(f"Noisy train images: {tuple(noisy_train_images.shape)}")
print(f"Denoised train images: {tuple(denoised_train_images.shape)}")
print(f"Clean test images: {tuple(clean_test_images.shape)}")
print(f"Noisy test images: {tuple(noisy_test_images.shape)}")
print(f"Denoised test images: {tuple(denoised_test_images.shape)}")
print(f"Noisy images saved to: {NOISY_OUTPUT_DIR}")
print(f"Denoised images saved to: {DENOISED_OUTPUT_DIR}")
print(f"Trained denoiser saved to: {MODEL_OUTPUT_PATH}")
print(f"Train: Loss={train_loss:.4f}, PSNR={train_psnr:.2f} dB, SSIM={train_ssim:.3f}")
print(f"Test: Loss={test_loss:.4f}, PSNR={test_psnr:.2f} dB, SSIM={test_ssim:.3f}")
print(
    "Noisy -> Clean: "
    f"Loss={noisy_test_loss:.4f}, "
    f"PSNR={noisy_test_psnr:.2f} dB, "
    f"SSIM={noisy_test_ssim:.3f}"
)
print(
    "Denoised -> Clean: "
    f"Loss={test_loss:.4f}, "
    f"PSNR={test_psnr:.2f} dB, "
    f"SSIM={test_ssim:.3f}"
)
print(f"Improvement: +{psnr_improvement:.2f} dB PSNR, +{ssim_improvement:.3f} SSIM")
