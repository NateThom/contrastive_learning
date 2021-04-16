import torch
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision
import copy

class Att_Dataset(Dataset):
    def __init__(self, fold, image_path, image_dir, attr_label_path, attributes_to_use, simclr_flag=False, transform=None):
        if fold == "testing":
            self.indices = list(range(160000))
        elif fold == "validation":
            self.indices = torch.tensor(list(range(160000, 160000 + 20000)))
        else:
            self.indices = torch.tensor(list(range(160000 + 20000, 160000+20000+20000)))

        self.img_path = image_path
        self.img_dir = image_dir

        # Read the binary attribute labels from the specified file
        self.img_labels = pd.read_csv(attr_label_path, sep=',', skiprows=0, usecols=attributes_to_use)

        # Get the paths to each of the input images
        self.input_filenames = pd.read_csv(attr_label_path, sep=',', skiprows=0, usecols=[0])

        self.simclr_flag = simclr_flag

        # If there are any transform functions to be called, store them
        self.transform = transform

    def __len__(self):
        return len(self.input_filenames)

    def __getitem__(self, idx):
        img_path = self.img_path + self.img_dir + self.input_filenames.iloc[idx, 0]
        image = torchvision.io.read_image(img_path)
        image = TF.convert_image_dtype(image, torch.float)

        # image = cv2.imread(img_path)
        # image = image.astype('uint8')
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = TF.to_tensor(image)

        # image = Image.open(img_path)

        # Read in the attribute labels for the current input image
        attributes = self.img_labels.iloc[idx,]
        attributes = torch.tensor(attributes)
        attributes = torch.gt(attributes, 0).float()

        if self.simclr_flag:
            image2 = copy.deepcopy(image)
            sample = {"image": image, "image2": image2, "label": attributes}
        else:
            sample = {"image": image, "label": attributes}

        if self.transform:
            sample = self.transform(sample)

        return sample
