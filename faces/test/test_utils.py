import unittest
from pathlib import Path

import PIL.Image

from faces.utils import preprocess


class TestUtils(unittest.TestCase):
    def test_preprocess(self):
        image = PIL.Image.open(
            Path(__file__).parent / "data" / "images" / "douglas_adams.jpg"
        )
        self.assertEqual(preprocess(image, 100).size, (100, 64))
        self.assertEqual(preprocess(image, 1000).size, (1000, 643))
        self.assertEqual(preprocess(image, 1000, rotate=90).size, (643, 1000))
        self.assertEqual(preprocess(image, 1000, rotate=180).size, (1000, 643))
        self.assertEqual(preprocess(image, 1000, rotate=270).size, (643, 1000))
        self.assertEqual(preprocess(image, 1000, rotate=360).size, (1000, 643))
        self.assertEqual(preprocess(image, 1000, rotate=450).size, (643, 1000))


if __name__ == "__main__":
    unittest.main()
