from abc import abstractmethod
from io import BytesIO
from pathlib import Path

import mlx.core as mx
import requests
from PIL import Image, ImageOps
from PIL.Image import Resampling as PILResampling
from transformers.image_processing_utils import BaseImageProcessor as ImageProcessor
from transformers.image_processing_utils import get_size_dict
from transformers.image_utils import ChannelDimension


class BaseImageProcessor(ImageProcessor):
    def __init__(
        self,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        size=(384, 384),
        crop_size: dict[str, int] | None = None,
        resample=PILResampling.BICUBIC,
        rescale_factor=1 / 255,
        data_format=ChannelDimension.FIRST,
    ):
        if not crop_size:
            crop_size = {"height": 448, "width": 448}
        crop_size = get_size_dict(
            crop_size,
            default_to_square=True,
            param_name="crop_size",
        )

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    @abstractmethod
    def preprocess(self, images: list[Image.Image]) -> list[mx.array]:
        pass

def load_image(image_source: str | Path | BytesIO, timeout: int = 10) -> Image.Image:
    """
    Helper function to load an image from either a URL or file.
    """
    if isinstance(image_source, BytesIO) or Path(image_source).is_file():
        # for base64 encoded images
        try:
            image = Image.open(image_source)
        except OSError as e:
            raise ValueError(
                f"Failed to load image from {image_source} with error: {e}"
            ) from e
    elif isinstance(image_source, str) and image_source.startswith(
        ("http://", "https://")
    ):
        try:
            response = requests.get(image_source, stream=True, timeout=timeout)
            response.raise_for_status()
            image = Image.open(response.raw)
        except Exception as e:
            raise ValueError(
                f"Failed to load image from URL: {image_source} with error {e}"
            ) from e
    else:
        raise ValueError(
            f"The image {image_source} must be a valid URL or existing file."
        )

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def resize_image(img: Image.Image, max_size: tuple[int, int]) -> Image.Image:
    ratio = min(max_size[0] / img.width, max_size[1] / img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    return img.resize(new_size)


def process_image(
    img: Image.Image | str,
    resize_shape: tuple[int, int] | None,
) -> Image.Image:
    if isinstance(img, str):
        img = load_image(img)

    if resize_shape is not None:
        img = resize_image(img, resize_shape)

    return img
