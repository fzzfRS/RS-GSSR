# Python
import os
import cv2
import numpy as np
from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset


class HEIalignedDataset(BaseDataset):
    """
    This dataset class loads aligned datasets for shadow removal tasks.

    It requires three directories:
    - Shadow images: '.../shadow'
    - Corresponding masks: '.../mask'
    - Shadow-free images: '.../shadowfree'

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
        self.dir_B = os.path.join(opt.dataroot, "shadowfree")  # Path to shadow-free images

        # Load image and mask paths
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.A_mask_paths = sorted(make_dataset(self.dir_A_mask, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))

        self.A_size = len(self.A_paths)  # Number of shadow images

        # Initialize transformations
        transform_params = get_params(opt, [opt.load_size, opt.load_size])
        self.transform_RGB = get_transform(opt, transform_params, grayscale=False)
        self.transform_Gray = get_transform(opt, transform_params, grayscale=True)

    def __getitem__(self, index):
        """Return a training pair and associated data.

        Args:
            index: Index of the dataset item.

        Returns:
            dict: Contains tensors for shadow image, shadow-free image, HMC processed data, and paths.
        """
        # Get paths for shadow image, mask, and shadow-free image
        A_path = self.A_paths[index]
        A_mask_path = self.A_mask_paths[index]
        B_path = self.B_paths[index]

        # Load images
        A_img = Image.open(A_path).convert("RGB")
        A_mask_img = Image.open(A_mask_path).convert("L")
        B_img = Image.open(B_path).convert("RGB")

        # Convert images to numpy arrays
        A_img_np = np.array(A_img)
        A_mask_np = np.array(A_mask_img)
        B_img_np = np.array(B_img)

        # Apply HEI processing
        DE, umbra, edge_mask, umbra_mask, shadowfree_masked = self.apply_hei_processing(
            A_img_np, A_mask_np, B_img_np, self.opt
        )

        # Convert processed images to PIL format
        DE_pil = Image.fromarray(DE).convert("L")
        umbra_pil = Image.fromarray(umbra)
        edge_mask_pil = Image.fromarray(edge_mask)
        umbra_mask_pil = Image.fromarray(umbra_mask)
        shadowfree_masked_pil = Image.fromarray(shadowfree_masked)

        # Apply transformations to images
        DE_tensor = self.transform_Gray(DE_pil)
        umbra_tensor = self.transform_RGB(umbra_pil)
        edge_mask_tensor = self.transform_Gray(edge_mask_pil)
        umbra_mask_tensor = self.transform_Gray(umbra_mask_pil)
        shadowfree_masked_tensor = self.transform_RGB(shadowfree_masked_pil)

        return {
            "de": DE_tensor,
            "umbra": umbra_tensor,
            "edge_mask": edge_mask_tensor,
            "umbra_mask": umbra_mask_tensor,
            "shadowfree": shadowfree_masked_tensor,
            "A_paths": A_path,
        }

    def __len__(self):
        """Return the size of the dataset."""
        return self.A_size

    def apply_hei_processing(self, shadow_img, mask, shadowfree, opt):
        """Apply HEI processing.

        Args:
            shadow_img: Numpy array of shadow image.
            mask: Numpy array of shadow mask.
            shadowfree: Numpy array of shadow-free image.
            opt: Options containing dataset configurations.

        Returns:
            Tuple of numpy arrays: Processed images including HEI data.
        """
        # Initialize output image
        out = shadow_img.copy()

        # Process mask
        mask = mask.copy() / 255.0
        mask[mask > 0] = 1

        for c in range(3):
            if opt.transfer2white:
                mask[shadow_img[:, :, c] == 255] = 0
            else:
                mask[shadow_img[:, :, c] == 0] = 0

        X2 = mask.astype(np.uint8)

        # Compute buffer regions
        se10 = np.ones((10, 10), np.uint8)
        se05 = np.ones((5, 5), np.uint8)
        sh_buff = cv2.dilate(X2, se10) & (~X2).astype(np.uint8)
        sd_buff = cv2.erode(X2, se05) & X2

        # Edge detection and dilation
        edges = cv2.Canny(X2 * 255, 100, 200)
        dilated_edges = cv2.dilate(edges, np.ones((4, 4), np.uint8), iterations=1)

        # Compute umbra mask
        umbra_mask = X2 - (dilated_edges > 0).astype(np.uint8)

        # Process each channel
        for k in range(3):
            img1 = shadow_img[:, :, k].astype(np.uint8)
            sh_region = img1[sd_buff == 1]
            if opt.transfer2white:
                buff_region = img1[(sh_buff == 1) & (img1 != 255)]
            else:
                buff_region = img1[(sh_buff == 1) & (img1 != 0)]

            if len(buff_region) > 0:
                # Compute histograms
                h_sh, _ = np.histogram(sh_region, bins=np.arange(257))
                h_buf, _ = np.histogram(buff_region, bins=np.arange(257))

                # Smooth and normalize histograms
                h_sh = self.smooth_histogram(h_sh)
                h_buf = self.smooth_histogram(h_buf)
                h_sh = h_sh / np.sum(h_sh)
                h_buf = h_buf / np.sum(h_buf)

                # Compute cumulative histograms
                c_sh = np.cumsum(h_sh)
                c_buf = np.cumsum(h_buf)

                # Build and smooth LUT
                x = np.arange(256)
                D = np.abs(c_sh[:, None] - c_buf[None, :])
                index = np.argmin(D, axis=1)
                lut = np.column_stack((x, index))
                lut = self.smooth_lut(lut)

                # Apply LUT
                img_out = img1.copy()
                tmp = img1[X2 == 1].astype(int)
                img_out[X2 == 1] = lut[tmp, 1]
                out[:, :, k] = img_out

        # Mask edge regions
        if opt.transfer2white:
            out[dilated_edges == 255] = 0
            umbra = shadow_img.copy()
            umbra[dilated_edges == 255] = 0
            shadowfree[dilated_edges == 255] = 0
        else:
            out[dilated_edges == 255] = 255
            umbra = shadow_img.copy()
            umbra[dilated_edges == 255] = 255
            shadowfree[dilated_edges == 255] = 255

        return (
            out.astype(np.uint8),
            umbra.astype(np.uint8),
            dilated_edges.astype(np.uint8),
            (umbra_mask * 255).astype(np.uint8),
            shadowfree.astype(np.uint8),
        )

    def smooth_histogram(self, hist, kernel_size=5):
        """Smooth histogram using a moving average filter.

        Args:
            hist: Numpy array of histogram values.
            kernel_size: Size of the smoothing kernel.

        Returns:
            Numpy array of smoothed histogram values.
        """
        smoothed_hist = np.copy(hist)
        for i in range(kernel_size, len(hist) - kernel_size):
            smoothed_hist[i] = np.mean(hist[i - kernel_size : i + kernel_size])
        return smoothed_hist

    def smooth_lut(self, lut, kernel_size=5):
        """Smooth LUT using a moving average filter.

        Args:
            lut: Numpy array of LUT values.
            kernel_size: Size of the smoothing kernel.

        Returns:
            Numpy array of smoothed LUT values.
        """
        smoothed_lut = np.copy(lut[:, 1])
        for i in range(kernel_size, len(lut) - kernel_size):
            smoothed_lut[i] = np.mean(lut[i - kernel_size : i + kernel_size, 1])
        lut[:, 1] = smoothed_lut
        return lut