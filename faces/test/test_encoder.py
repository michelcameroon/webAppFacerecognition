import unittest
from os.path import basename
from pathlib import Path

import numpy as np
import torch

from faces import FaceEncoding, FacePatch
from faces.encoder import ResnetEncoder


class TestEncoder(unittest.TestCase):
    def setUp(self) -> None:
        self.encoder = ResnetEncoder(torch.device("cpu"))

    def test_call(self) -> None:
        # test dimension on a single patch
        patch = FacePatch(
            np.load(Path(__file__).parent / "data" / "patches" / "graham-chapman.npy")
        )
        encoding = self.encoder(patch)
        self.assertIsInstance(encoding, FaceEncoding)
        self.assertEqual(encoding.shape, (512,))

        # test multiple patches
        for query in (
            "eric-idle.npy",
            "graham-chapman.npy",
            "john-cleese.npy",
            "michael-palin.npy",
            "terry-gilliam.npy",
            "terry-jones.npy",
        ):
            np.testing.assert_array_equal(
                self.encoder(
                    FacePatch(
                        np.load(Path(__file__).parent / "data" / "patches" / query)
                    )
                )
                .detach()
                .cpu()
                .numpy(),
                np.load(Path(__file__).parent / "data" / "encodings" / query),
            )


if __name__ == "__main__":
    unittest.main()
