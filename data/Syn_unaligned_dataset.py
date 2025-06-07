import os
import random

import cv2
import numpy as np
from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset


class SynUnalignedDataset(BaseDataset):
    """
    This dataset class loads synthetic unaligned/unpaired datasets.

    It requires two directories:
    - Shadow images: '.../Shadow'
    - Corresponding masks: '.../Mask'

    The dataset flag '--dataroot /path/to/data' specifies the root directory.
    """

    def __init__(self, opt):
        """Initialize the dataset with paths and transformations.

        Args:
            opt: Options containing dataset configurations.
        """
        super().__init__(opt)
        self.dir_A = os.path.join(opt.dataroot, "shadow")  # Path to shadow images
        self.dir_A_mask = os.path.join(opt.dataroot, "mask")  # Path to shadow masks

        # Load image and mask paths
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.A_mask_paths = sorted(make_dataset(self.dir_A_mask, opt.max_dataset_size))

        self.A_size = len(self.A_paths)  # Number of shadow images
        self.custom_size = opt.custom_dataset_size  # Custom dataset size

        # Initialize transformations
        transform_params = get_params(opt, [opt.load_size, opt.load_size])
        self.transform_RGB = get_transform(opt, transform_params, grayscale=False)
        self.transform_Gray = get_transform(opt, transform_params, grayscale=True)

    def __getitem__(self, index):
        """Return a training pair and associated data.

        Args:
            index: Index of the dataset item.

        Returns:
            dict: Contains tensors for shadow image, shadow-free image, matte mask, real shadow image, and paths.
        """
        # Randomly select an index for shadow image and mask
        random_index = random.randint(0, self.A_size - 1)
        A_path = self.A_paths[random_index]
        A_mask_path = self.A_mask_paths[random_index]
        A_img = Image.open(A_path).convert("RGB")
        A_mask = Image.open(A_mask_path).convert("L")

        # Randomly select another image and mask for domain B
        index_B = self._get_random_index(index)
        B_path = self.A_paths[index_B]
        B_mask_path = self.A_mask_paths[index_B]
        B_img = Image.open(B_path).convert("RGB")
        B_mask = Image.open(B_mask_path).convert("L")

        # Perform random cropping
        A_img, A_mask, B_img, B_mask = self.random_crop(
            [A_img, A_mask, B_img, B_mask], self.opt
        )

        # Convert images and masks to numpy arrays
        A_img_np = np.array(A_img)
        A_mask_np = np.array(A_mask)

        # Apply guided filtering to refine the mask
        A_mask_filtered = cv2.ximgproc.guidedFilter(
            guide=A_img_np, src=A_mask_np, radius=3, eps=1e-4, dDepth=-1
        )

        # Set masked pixels in the shadow image to black
        A_img_np[A_mask_filtered > 125] = [0, 0, 0]

        # Sample darkening parameters for shadow synthesis
        b1_R, w1_R, b1_G, w1_G, b1_B, w1_B = self.sample_darkening_params_UAV(self.opt)

        # Normalize and apply darkening to the shadow image
        A_img_np = A_img_np / 255.0
        a = np.array([w1_R, w1_G, w1_B]).reshape(1, 1, 3)
        b = np.array([b1_R, b1_G, b1_B]).reshape(1, 1, 3)
        A_img_np_dark = np.clip(a * A_img_np + b, 0, 1)

        # Process domain B image and mask
        B_img_np = np.array(B_img)
        B_mask_np = np.array(B_mask)
        B_mask_np_filtered = (
            cv2.ximgproc.guidedFilter(
                guide=B_img_np, src=B_mask_np, radius=3, eps=1e-4, dDepth=-1
            )
            / 255.0
        )
        B_mask_np_filtered[A_mask_filtered > 125] = 0

        # Binarize the matte mask
        B_matte_np = (B_mask_np_filtered > 0.5).astype(np.uint8)
        B_matte_np_255 = (B_matte_np * 255).astype(np.uint8)
        B_matte_np_pil = Image.fromarray(B_matte_np_255)

        # Expand filtered mask for broadcasting
        B_mask_np_filtered_expand = np.expand_dims(B_mask_np_filtered, axis=-1)

        # Generate initial shadow image
        ini_shadow = (
            A_img_np * (1 - B_mask_np_filtered_expand)
            + A_img_np_dark * B_mask_np_filtered_expand
        )
        ini_shadow[np.logical_or(A_mask_filtered > 125, B_matte_np == 0)] = [0, 0, 0]
        ini_shadow = (np.clip(ini_shadow, 0, 1) * 255).astype(np.uint8)
        ini_shadow_pil = Image.fromarray(ini_shadow)

        # Generate initial shadow-free image
        ini_shadowfree = A_img_np.copy()
        ini_shadowfree[A_mask_filtered > 125] = [0, 0, 0]
        ini_shadowfree = (np.clip(ini_shadowfree, 0, 1) * 255).astype(np.uint8)
        ini_shadowfree_pil = Image.fromarray(ini_shadowfree)

        # Process real shadow image
        B_img_np[np.logical_or(A_mask_filtered > 125, B_matte_np == 0)] = [0, 0, 0]
        B_img_pil = Image.fromarray(B_img_np.astype(np.uint8))

        # Apply transformations to images and masks
        ini_shadow_tensor = self.transform_RGB(ini_shadow_pil)
        ini_shadowfree_tensor = self.transform_RGB(ini_shadowfree_pil)
        matte_tensor = self.transform_Gray(B_matte_np_pil)
        real_shadow_tensor = self.transform_RGB(B_img_pil)

        # Extract filenames and extension
        A_filename = os.path.splitext(os.path.basename(A_path))[0]
        B_filename, B_ext = os.path.splitext(os.path.basename(B_path))

        # Create combined filename with counter and original extension
        combined_filename = f"{index + 1}_{A_filename}&{B_filename}{B_ext}"

        # Create full combined path
        AB_path = os.path.join(os.path.dirname(A_path), combined_filename)

        return {
            "A": ini_shadow_tensor,
            "A_free": ini_shadowfree_tensor,
            "matte": matte_tensor,
            "B": real_shadow_tensor,
            "AB_paths": AB_path,
        }

    def __len__(self):
        """Return the custom dataset size."""
        return self.custom_size

    def sample_darkening_params_UAV(self, opt):
        """
        Sample darkening parameters from a predefined parameter file.

        Args:
            opt: Options containing dataset configurations.

        Returns:
            Tuple of six floats: b1_R, w1_R, b1_G, w1_G, b1_B, w1_B
        """
        params_path = os.path.join(os.path.dirname(opt.dataroot), "parameters.txt")
        with open(params_path, "r") as f:
            lines = f.readlines()
        line = random.choice(lines)
        b1_R, w1_R, b1_G, w1_G, b1_B, w1_B = map(float, line.strip().split())
        return b1_R, w1_R, b1_G, w1_G, b1_B, w1_B

    def _get_random_index(self, index):
        """
        Get a random index for domain B.

        Args:
            index (int): Current index.

        Returns:
            int: Random index for domain B.
        """
        if self.opt.serial_batches:
            return (index + 1) % self.A_size
        else:
            index_B = random.randint(0, self.A_size - 1)
            while index_B == index:
                index_B = random.randint(0, self.A_size - 1)
            return index_B

    def random_crop(self, images, opt):
        """
        Perform a random crop on a list of images with the same crop position.

        Args:
            images (list of PIL.Image): List of images to crop.
            opt: Options containing crop size.

        Returns:
            list of PIL.Image: Cropped images.
        """
        # Get the dimensions of the first image
        width, height = images[0].size

        crop_width, crop_height = (
            opt.syn_crop_size,
            opt.syn_crop_size,
        )  # Assuming square crops

        # Ensure the crop size is valid
        if crop_width > width or crop_height > height:
            raise ValueError("Crop size must be smaller than the image dimensions.")

        # Randomly select the top-left corner for the crop
        x = random.randint(0, width - crop_width)
        y = random.randint(0, height - crop_height)

        # Crop all images using the same position
        cropped_images = [
            img.crop((x, y, x + crop_width, y + crop_height)) for img in images
        ]

        return cropped_images