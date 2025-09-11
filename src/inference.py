import matplotlib.pyplot as plt
import albumentations as A
import sys
import torch

from dataset import BuildingDataset
from model import SegmentationModel


import matplotlib.pyplot as plt


def visualise(**images):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), constrained_layout=True)

    if n == 1:
        axes = [axes]

    for ax, (name, image) in zip(axes, images.items()):
        ax.axis("off")
        ax.set_title(" ".join(name.split("_")).title(), fontsize=12, pad=8)

        if name == "image":
            ax.imshow(image.permute(1, 2, 0))
        else:
            ax.imshow(image.squeeze(), cmap="gray")

    plt.savefig(
        f"./assets/inference_{IMAGE_INDEX}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)


DATA_PATH = "archive/png/"
# Get IMAGE_INDEX from CLI, defaults to 0
try:
    IMAGE_INDEX = int(sys.argv[1])
except (IndexError, ValueError):
    print("[DEBUG] No image index provided. Using default value: 0")
    IMAGE_INDEX = 0

CHECKPOINT_PATH = "checkpoints/EfficientNet-B3-trained.pth"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.mps.is_available() else "cpu"
)
THRESHOLD = 0.5

test_dataset = BuildingDataset(
    img_dir=DATA_PATH + "/test",
    mask_dir=DATA_PATH + "/test_labels",
    transform=A.ToTensorV2(),
)

print("[INFO] Loading model...")
model = SegmentationModel.load_from_checkpoint(CHECKPOINT_PATH)
model.eval()
model.to(DEVICE)

print(f"[INFO] Loading mask and image...")

image, mask = test_dataset[IMAGE_INDEX]

with torch.no_grad():
    image, mask = image.to(DEVICE), mask.to(DEVICE)

    logits = model(image)

    probs = torch.sigmoid(logits)
    pred = (probs > THRESHOLD).float()


visualise(image=image.cpu(), mask=mask.cpu(), prediction=pred.cpu())
