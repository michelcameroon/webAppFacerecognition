import unittest
from pathlib import Path

from PIL import Image as PILImage

from faces import BoundingBox, Image
from faces.drawing import PILAnnotate


class TestAnnotate(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.open(
            Path(__file__).parent / "data" / "images" / "douglas_adams.jpg"
        )
        self.annotate = PILAnnotate()
        self.bounding_boxes = [
            BoundingBox(10, 20, 30, 40),
            BoundingBox(40, 10, 60, 20),
            BoundingBox(80, 50, 160, 60),
        ]

    def test_with_probability(self) -> None:
        annotated_image = self.annotate.with_probability(
            self.image,
            ((box, index / 10) for index, box in enumerate(self.bounding_boxes)),
        )
        self.assertIsInstance(annotated_image, PILImage.Image)
        self.assertEqual(annotated_image.size, self.image.image.size)

    def test_with_identity(self) -> None:
        annotated_image = self.annotate.with_identity(
            self.image,
            ((box, "Hello world") for index, box in enumerate(self.bounding_boxes)),
        )
        self.assertIsInstance(annotated_image, PILImage.Image)
        self.assertEqual(annotated_image.size, self.image.image.size)

    def test_with_enumeration(self) -> None:
        annotated_image = self.annotate.with_enumeration(
            self.image, self.bounding_boxes
        )
        self.assertIsInstance(annotated_image, PILImage.Image)
        self.assertEqual(annotated_image.size, self.image.image.size)

    def test_call(self) -> None:
        annotated_image = self.annotate(self.image, self.bounding_boxes)
        self.assertIsInstance(annotated_image, PILImage.Image)
        self.assertEqual(annotated_image.size, self.image.image.size)


if __name__ == "__main__":
    unittest.main()
