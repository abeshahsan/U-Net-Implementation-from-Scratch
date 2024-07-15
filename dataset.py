import PIL
import PIL.Image
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np


class UnetDataSet(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = []
        self.mask_paths = []
        self.transform = transform

        if isinstance(image_dir, str):
            image_dir = [image_dir]
        if isinstance(mask_dir, str):
            mask_dir = [mask_dir]

        for i in range(len(image_dir)):
            assert os.path.exists(image_dir[i]), \
                f"Image directory {image_dir[i]} does not exist"

        for i in range(len(mask_dir)):
            assert os.path.exists(mask_dir[i]), \
                f"Target directory {mask_dir[i]} does not exist"

        for i in range(len(image_dir)):
            len_of_images = len(os.listdir(image_dir[i]))
            len_of_masks = len(os.listdir(mask_dir[i]))

            assert len_of_images == len_of_masks, (
                "Number of images and targets should be same for "
                f"{image_dir[i]} and {mask_dir[i]}"
            )

        for i in range(len(image_dir)):
            self.image_paths.extend(
                [
                    os.path.join(image_dir[i], image_path)
                    for image_path in os.listdir(image_dir[i])
                ]
            )
            self.mask_paths.extend(
                [
                    os.path.join(mask_dir[i], mask_path)
                    for mask_path in os.listdir(mask_dir[i])
                ]
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx])
                        .convert("L"), dtype=np.float32)

        if self.transform:
            transformation = self.transform(image=image, mask=mask)
            image = transformation["image"]
            mask = transformation["mask"]

        image /= 255.0

        return image, mask


class CityScapesDataset(Dataset):
    """
        A dataset class for the CityScapes dataset.
    """

    def __init__(self, data_dir, transform=None):
        """
        Initialize the dataset.

        Parameters:
        ----------
        `data_dir` : str
            The path to the directory containing the images.
            `transform` : albumentations.Compose
            A composition of image transformations to apply to each image

        Returns:
        --------
        None
        """

        self.image_dir = data_dir
        self.transform = transform
        self.image_paths = os.listdir(data_dir)

    def __len__(self):
        """
        Get the number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get an image and its corresponding target mask from the dataset.

        Parameters:
        ----------
        `idx` : int
            The index of the image to retrieve.

        Returns:
        --------
        `image` : np.ndarray
            The image.
        """
        image_path = os.path.join(self.image_dir, self.image_paths[idx])

        image = Image.open(image_path).convert("RGB")

        image, target = self.__split_image__(image)

        image = np.array(image.convert("RGB"))
        target = np.array(target.convert("L"), dtype=np.float32)
        # target = cityscapes_labels_map(target)

        if self.transform:
            transformation = self.transform(image=image, mask=target)
            image = transformation["image"]
            target = transformation["mask"]

        return image, target

    def __split_image__(self, image: PIL.Image):
        """
        Split an image into left and right halves.

        Parameters:
        ----------
        `image` : PIL.Image
            The image to split.

        Returns:
        --------
        `left_image` : PIL.Image
            The left half of the image.
        `right_image` : PIL.Image
            The right half of the image.
        """
        width, height = image.size  # Get the width and height of the image

        # Calculate the middle of the image
        middle = width // 2

        # Split the image into left and right halves
        left_image = image.crop((0, 0, middle, height))
        right_image = image.crop((middle, 0, width, height))

        return left_image, right_image
