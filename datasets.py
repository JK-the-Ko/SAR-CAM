from os import listdir
from os.path import join

import random
from random import randint
import numpy as np

import PIL.Image as pil_image

from torch.utils.data import Dataset
import torchvision.transforms as transforms

# From clean Image
class DataFromFolder(Dataset) :
    def __init__(self, noisy_image_dir, clean_image_dir, mode, seed) :
        # Inheritance
        super(DataFromFolder, self).__init__()

        # Fix Seed
        random.seed(seed)
        np.random.seed(seed)

        # Create Path List Instance
        self.noisy_image_path_list = [join(noisy_image_dir, image) for image in listdir(noisy_image_dir)]
        self.clean_image_path_list = [join(clean_image_dir, image) for image in listdir(clean_image_dir)]

        # Sort Path List in Order
        self.noisy_image_path_list.sort()
        self.clean_image_path_list.sort()

        # Get Probability
        p_h = randint(0, 1)
        p_v = randint(0, 1)

        # Data Augmentation
        if mode == "train" :
            self.transform = transforms.Compose([
                                                            transforms.RandomHorizontalFlip(p = p_h),
                                                            transforms.RandomVerticalFlip(p = p_v),
                                                            transforms.ToTensor()]
                                                            )

        elif mode == "valid" :
            self.transform = transforms.ToTensor()

    def _load_image_(self, image_path) :
        # Convert into Pillow Image
        image = pil_image.open(image_path)

        return image

    def __getitem__(self, index) :
        # Load Image
        input = self._load_image_(self.noisy_image_path_list[index])
        target = self._load_image_(self.clean_image_path_list[index])

        # Apply PyTorch Transforms
        input = self.transform(input)
        target = self.transform(target)

        return input, target

    def __len__(self) :
        # Get Number of Data
        return len(self.clean_image_path_list)