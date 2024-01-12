import shutil
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import mkstemp

import torch
from PIL import Image as PILImage

from faces import Image
from faces.builder import DefaultBuilder
from faces.main import Main


class TestMain(unittest.TestCase):
    def setUp(self) -> None:
        self.registry_path = Path(mkstemp(prefix="faces-test-")[1])
        shutil.copy(
            Path(__file__).parent / "data" / "registry" / "faces.pkl",
            self.registry_path,
        )
        self.builder = DefaultBuilder(
            device=torch.device("cpu"),
            registry_path=self.registry_path,
        )

    def tearDown(self) -> None:
        self.registry_path.unlink(missing_ok=True)

    def test_detect(self) -> None:
        image = Image.open(
            Path(__file__).parent / "data" / "images" / "douglas_adams.jpg"
        )
        annotated_image = Main().detect(self.builder, image)
        self.assertIsInstance(annotated_image, PILImage.Image)

    def test_detect_with_probability(self) -> None:
        image = Image.open(
            Path(__file__).parent / "data" / "images" / "douglas_adams.jpg"
        )
        annotated_image = Main().detect_with_probability(self.builder, image)
        self.assertIsInstance(annotated_image, PILImage.Image)

    def test_identify(self) -> None:
        image = Image.open(
            Path(__file__).parent / "data" / "images" / "douglas_adams.jpg"
        )
        annotated_image = Main().identify(self.builder, image)
        self.assertIsInstance(annotated_image, PILImage.Image)

    def test_register(self) -> None:
        self.assertEqual(len(self.builder.registry), 4)
        Main().register(
            self.builder,
            Path(__file__).parent / "data" / "images" / "douglas_adams.jpg",
        )
        self.assertEqual(len(self.builder.registry), 5)

    def test_dump(self) -> None:
        buffer = StringIO()
        with redirect_stdout(buffer):
            Main().dump(self.builder)
        self.assertSetEqual(
            {line.strip() for line in buffer.getvalue().split("\n") if line.strip()},
            {
                "1: terry-jones.npy",
                "1: michael-palin.npy",
                "1: terry-gilliam.npy",
                "1: john-cleese.npy",
            },
        )

    def test_remove(self) -> None:
        self.assertEqual(len(self.builder.registry), 4)
        Main().remove(self.builder, "terry-jones.npy")
        Main().remove(self.builder, "terry-gilliam.npy")
        Main().remove(self.builder, "not present")
        self.assertEqual(len(self.builder.registry), 2)


if __name__ == "__main__":
    unittest.main()
