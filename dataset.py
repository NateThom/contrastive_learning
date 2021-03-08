import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class Att_Dataset(Dataset):
    def __init__(self, fold, image_path, image_dir, attr_label_path, attributes_to_use, transform=None):
        if fold == "testing":
            self.indices = list(range(160000))
        elif fold == "validation":
            self.indices = torch.tensor(list(range(160000, 160000 + 20000)))
        else:
            self.indices = torch.tensor(list(range(160000 + 20000, 160000+20000+20000)))

        self.img_path = image_path
        self.img_dir = image_dir

        # Read the binary attribute labels from the specified file
        self.attr_labels = pd.read_csv(attr_label_path, sep=',', skiprows=0, usecols=attributes_to_use)
        self.img_labels = pd.read_csv(attr_label_path, sep=',', skiprows=0, usecols=[n for n in range(1, 6)])

        # Get the paths to each of the input images
        self.input_filenames = pd.read_csv(attr_label_path, sep=',', skiprows=0, usecols=[0])

        # If there are any transform functions to be called, store them
        self.transform = transform

    def __len__(self):
        return len(self.input_filenames)

    def __getitem__(self, idx):
        img_path = self.img_path + self.img_dir + self.input_filenames.iloc[idx, 0]
        image = read_image(img_path)
        temp = self.input_filenames.iloc[idx, 0]

        # Read in the attribute labels for the current input image
        attributes = self.img_labels.iloc[idx,]
        attributes = torch.tensor(attributes)
        attributes = torch.gt(attributes, 0)

        sample = {"image": image.float(), "label": attributes.int()}
        if self.transform:
            sample = self.transform(sample)


        return sample