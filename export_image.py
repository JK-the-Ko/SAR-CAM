import numpy as np

from torchvision import transforms

def concat_image(tensor_input, tensor_pred, tensor_target) :
        # Create Torchvision Transforms Instance
        to_pil = transforms.Compose([transforms.ToPILImage()])

        # Convert PyTorch Tensor to Numpy Array
        image_input = np.array(to_pil(tensor_input), dtype = "uint8")
        image_pred = np.array(to_pil(tensor_pred), dtype = "uint8")
        image_target = np.array(to_pil(tensor_target), dtype = "uint8")

        # Stack Images
        stacked_image = np.hstack((image_input, image_pred, image_target))

        return stacked_image