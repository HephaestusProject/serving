import os
import sys
import base64
import torch
import torchvision

from typing import Dict
from io import BytesIO
from PIL import Image

if sys:
    sys.path.insert(0, os.path.dirname(
        os.path.abspath(os.path.dirname(__file__))))
    from src.model.net import MNIST


class Handler(object):

    def __init__(self):
        root_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
        weight_file = os.path.join(root_path, "serving/mnist_weight.pt")
        self.model = MNIST()
        self.model.load_state_dict(torch.load(weight_file))

    def __call__(self, data: Dict):
        base64str = data["base64_image"]

        inputs = self._preprocessing(base64str)
        pred = self.model.inference(inputs)

        return pred

    @staticmethod
    def _preprocessing(base64str: str):
        image = Image.open(BytesIO(base64.b64decode(base64str)))
        inputs = torchvision.transforms.ToTensor()(image).unsqueeze(0)

        return inputs


if __name__ == "__main__":
    base64str = "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAAjklEQVR4nM3OIQrCcBiG8b+MMcyaZIdYWREvYBCxeAdZ2CEUDyEogpgtHmFNkx7AYBNcFhkPnuAxGXzrj+fjC+HfN6KIzDp3aBtOYN8SS04wtDCHt76zhKNixSsz68NTwxIWijvq1GzQcNNwDKtvV3OztOGi4RTmijMeXcUD59gsvlJpGG3YKobeunD84T5n9jYgy+eQ7wAAAABJRU5ErkJggg=="
    inputs = {"base64_image": base64str}
    handler = Handler()
    pred = handler(inputs)
    print(pred)
