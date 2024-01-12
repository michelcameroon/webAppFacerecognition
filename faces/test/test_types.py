import unittest
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from faces import BoundingBox, Image


class TestImage(unittest.TestCase):
    def test_from_array(self) -> None:
        image = Image.from_array(
            np.array(
                PILImage.open(Path(__file__).parent / "data" / "images" / "cactus.jpg")
            )
        )
        self.assertEqual(image.image.size, (1000, 664))

    def test_open(self) -> None:
        image = Image.open(Path(__file__).parent / "data" / "images" / "cactus.jpg")
        self.assertEqual(image.image.size, (1000, 664))


class TestBoundingBox(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
