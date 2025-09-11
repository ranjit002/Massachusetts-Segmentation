import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset


class BuildingDataset(Dataset):
    """Dataset for Massachusetts building segmentation tasks.

    This dataset loads images and corresponding segmentation masks for binary
    semantic segmentation tasks, such as detecting buildings in aerial images.

    Attributes:
        img_dir (str): Directory containing the input images.
        mask_dir (str): Directory containing the corresponding binary masks.
        transform (callable, optional):

    Notes:
        - Images loaded as RGB and scaled to the [0,1] range.
        - Masks loaded as grayscale (single channel) and scaled to [0,1].
        - The mask is expanded to have a shape `(1, H, W)`
    """

    def __init__(self, img_dir, mask_dir, transform=None):
        """
        Args:
            img_dir (str): Path to the image directory
            mask_dir (str): Path to the masks directory
            transform (callable, optional): Optional Albumentations transformation
                for data augmentation (e.g., random flips, crops).
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.imgs = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        """Retrieves image-mask pair

        Args:
            idx (int): Index of sample to retrieve.

        Returns:
            tuple: (image, mask) where:
                - image (torch.FloatTensor): Normalized RGB image tensor with format: (C, H, W)) and values [0,1]
                - mask (torch.FloatTensor): Binary mask tensor with format (1, H, W) and values [0,1]
        """
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # (H, W) -> (1, H, W)
        mask = mask[np.newaxis, ...]

        return image.float(), mask.float()
