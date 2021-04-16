import torch
import torchvision.transforms.functional as TF

class MyRandomBlur(object):
    """Crop randomly the image and masks in a sample
            output_size (tuple or int): Desired output size. If int, square crop
                is made.
        """

    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, sample):
        image = sample["image"]

        blur_probability = torch.rand(1)

        if blur_probability[0] > 0.5:
            image = TF.gaussian_blur(image, self.kernel_size)

        if "image2" in sample:
            image2 = sample["image2"]

            blur_probability = torch.rand(1)

            if blur_probability[0] > 0.5:
                image2 = TF.gaussian_blur(image2, self.kernel_size)

            return {'image': image, "image2": image2, 'label': sample['label']}

        return {'image': image, 'label': sample['label']}