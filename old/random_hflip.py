import torch
import torchvision.transforms.functional as TF

class MyRandomHFlip(object):
    """Crop randomly the image and masks in a sample
            output_size (tuple or int): Desired output size. If int, square crop
                is made.
        """

    def __call__(self, sample):
        image = sample["image"]

        flip_probability = torch.rand(1)

        if flip_probability[0] > 0.5:
            image = TF.hflip(image)

        if "image2" in sample:
            image2 = sample["image2"]

            flip_probability = torch.rand(1)

            if flip_probability[0] > 0.5:
                image2 = TF.hflip(image2)

            return {'image': image, 'image2': image2, 'label': sample['label']}

        return {'image': image, 'label': sample['label']}