import torch

import torchvision.transforms.functional as TF

class MyRandomResizedCrop(object):
    """Crop randomly the image and masks in a sample
            output_size (tuple or int): Desired output size. If int, square crop
                is made.
        """

    def __init__(self, output_size):
        # Ensure that the arguments passed in are of the expected format
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample["image"]

        # Get the height and width of the image in sample
        image_h, image_w = image.shape[1:3]

        # Copy the height and width from output size
        new_image_h, new_image_w = self.output_size

        # Randomly select a point to crop the top and left edge of the image to
        top = torch.randint(0, image_h - new_image_h, (1,))
        left = torch.randint(0, image_w - new_image_w, (1,))

        image = TF.resized_crop(image, top[0], left[0], new_image_h, new_image_w, self.output_size)

        return {'image': image, 'attributes': sample['label']}