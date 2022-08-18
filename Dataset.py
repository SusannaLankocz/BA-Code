import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class AbstractPortraitDataset(Dataset):
    def __init__(self, root_abstract, root_portrait, transform=None):
        self.root_abstract = root_abstract
        self.root_portrait = root_portrait
        self.transform = transform

        self.abstract_images = os.listdir(root_abstract)
        self.portrait_images = os.listdir(root_portrait)
        self.length_dataset = max(len(self.abstract_images), len(self.portrait_images))
        self.abstract_len = len(self.abstract_images)
        self.portrait_len = len(self.portrait_images)


    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        abstract_img = self.abstract_images[index % self.abstract_len]  # no index error, correct index
        portrait_img = self.portrait_images[index % self.portrait_len]  # no index error, correct index

        abstract_path = os.path.join(self.root_abstract, abstract_img)
        portrait_path = os.path.join(self.root_portrait, portrait_img)

        # convert to numpy arrays
        abstract_img = np.array(Image.open(abstract_path).convert("RGB"))
        portrait_img = np.array(Image.open(portrait_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=abstract_img, image0=portrait_img)
            abstract_img = augmentations["image"]
            portrait_img = augmentations["image0"]

        return abstract_img, portrait_img

