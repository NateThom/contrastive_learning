import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T


class MyRandomColorJitter(object):
    """Crop randomly the image and masks in a sample
            output_size (tuple or int): Desired output size. If int, square crop
                is made.
        """

    def __init__(self):
        self.jitter = T.ColorJitter(0.8, 0.8, 0.8, 0.2)


    def __call__(self, sample):
        image = sample["image"]

        jitter_probability = torch.rand(1)

        if jitter_probability[0] > 0.5:
            image = self.jitter(image)

        if "image2" in sample:
            image2 = sample["image2"]

            jitter_probability = torch.rand(1)

            if jitter_probability[0] > 0.5:
                image2 = self.jitter(image2)

            return {'image': image, 'image2': image2, 'label': sample['label']}

        return {'image': image, 'label': sample['label']}
