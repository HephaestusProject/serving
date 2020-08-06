import base64
import torch
import torchvision

from pydantic import BaseModel
from pathlib import Path
from typing import Dict
from io import BytesIO
from PIL import Image

from src.model.net import MNIST


class Request(BaseModel):
    base64_image_string: str


class Response(BaseModel):
    prediction: str


class Handler(object):

    def __init__(self):
        """
        instantiation deep learning model 

        1. declare weight path
        2. instantiation deep learning model
        3. load weight and
        """
        root_path = Path(".").absolute()
        weight_file = root_path / "mnist_weight.pt"

        self.model: MNIST = MNIST()
        self.model.load_state_dict(torch.load(weight_file))

    def __call__(self, request: Request) -> str:
        """
        inference

        Args:
            request (Request): 딥러닝 모델 추론을 위한 입력 데이터

        Returns:
            (Response): 딥러닝 모델 추론 결과

        """
        base64image: str = request.base64_image_string
        inputs: torch.Tensor = self._preprocessing(base64image)
        prediction: str = self.model.inference(inputs)

        return Response(prediction=prediction)

    @staticmethod
    def _preprocessing(base64image: str) -> torch.Tensor:
        """
        base64로 encoding된 image를 torch.tensor로 변환

        Args:
            base64image (str): base64로 encoding된 이미지

        Returns:
            (torch.Tensor)
        """
        image = Image.open(BytesIO(base64.b64decode(base64image)))
        inputs = torchvision.transforms.ToTensor()(image).unsqueeze(0)

        return inputs


if __name__ == "__main__":
    base64str = "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAAjklEQVR4nM3OIQrCcBiG8b+MMcyaZIdYWREvYBCxeAdZ2CEUDyEogpgtHmFNkx7AYBNcFhkPnuAxGXzrj+fjC+HfN6KIzDp3aBtOYN8SS04wtDCHt76zhKNixSsz68NTwxIWijvq1GzQcNNwDKtvV3OztOGi4RTmijMeXcUD59gsvlJpGG3YKobeunD84T5n9jYgy+eQ7wAAAABJRU5ErkJggg=="
    inputs = {"base64_image_string": base64str}
    handler = Handler()
    predidction = handler(inputs)
    print(predidction)
