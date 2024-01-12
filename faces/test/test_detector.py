import unittest
from pathlib import Path

import numpy as np
import torch

from faces import BoundingBox, Image
from faces.detector import MTCNNDetector


class TestDetector(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = MTCNNDetector(
            device=torch.device("cpu"),
            probability_threshold=0.0,
            min_face_size=20,
            thresholds=(0.6, 0.7, 0.7),
            factor=0.709,
        )

    def test_detect_threshold(self) -> None:
        detector = MTCNNDetector(
            device=torch.device("cpu"),
            probability_threshold=1.0,
            min_face_size=20,
            thresholds=(0.6, 0.7, 0.7),
            factor=0.709,
        )
        self.assertFalse(
            set(
                detector.detect(
                    Image.open(
                        Path(__file__).parent / "data" / "images" / "douglas_adams.jpg"
                    )
                )
            )
        )
        self.assertFalse(
            set(
                detector.detect(
                    Image.open(
                        Path(__file__).parent / "data" / "images" / "monty_python.jpg"
                    )
                )
            )
        )

    def test_detect(self) -> None:
        self.assertSetEqual(
            set(
                self.detector.detect(
                    Image.open(
                        Path(__file__).parent / "data" / "images" / "douglas_adams.jpg"
                    )
                )
            ),
            {
                (  # douglas adams
                    BoundingBox(
                        lower_left=516.08984375,
                        lower_top=21.717365264892578,
                        upper_left=888.020751953125,
                        upper_top=546.5032348632812,
                    ),
                    0.999893069267273,
                )
            },
        )
        self.assertSetEqual(
            set(
                self.detector.detect(
                    Image.open(
                        Path(__file__).parent / "data" / "images" / "monty_python.jpg"
                    )
                )
            ),
            {
                (  # John Cleese
                    BoundingBox(
                        lower_left=786.6824340820312,
                        lower_top=48.136497497558594,
                        upper_left=875.5492553710938,
                        upper_top=161.416748046875,
                    ),
                    0.9998730421066284,
                ),
                (  # Graham Chapman
                    BoundingBox(
                        lower_left=659.3766479492188,
                        lower_top=183.70761108398438,
                        upper_left=747.95849609375,
                        upper_top=296.74285888671875,
                    ),
                    0.9951884746551514,
                ),
                (  # false positive
                    BoundingBox(
                        lower_left=657.21923828125,
                        lower_top=140.8580322265625,
                        upper_left=677.8623657226562,
                        upper_top=164.8173828125,
                    ),
                    0.907974362373352,
                ),
                (  # Terry Gilliam
                    BoundingBox(
                        lower_left=466.7940979003906,
                        lower_top=378.8565673828125,
                        upper_left=570.8778686523438,
                        upper_top=503.1236267089844,
                    ),
                    0.9999799728393555,
                ),
                (  # Terry Jones
                    BoundingBox(
                        lower_left=40.002967834472656,
                        lower_top=14.119751930236816,
                        upper_left=135.34446716308594,
                        upper_top=135.5555877685547,
                    ),
                    0.9943884015083313,
                ),
                (  # Eric Idle
                    BoundingBox(
                        lower_left=250.31846618652344,
                        lower_top=87.12207794189453,
                        upper_left=339.7062683105469,
                        upper_top=206.8123321533203,
                    ),
                    0.9997848868370056,
                ),
                (  # false positive
                    BoundingBox(
                        lower_left=738.0390014648438,
                        lower_top=527.5866088867188,
                        upper_left=759.5291748046875,
                        upper_top=555.6311645507812,
                    ),
                    0.7637770771980286,
                ),
                (  # Michael Palin
                    BoundingBox(
                        lower_left=485.0389709472656,
                        lower_top=154.45462036132812,
                        upper_left=582.5718383789062,
                        upper_top=279.035400390625,
                    ),
                    0.9971151351928711,
                ),
            },
        )
        self.assertFalse(
            list(
                self.detector.detect(
                    Image.open(Path(__file__).parent / "data" / "images" / "cactus.jpg")
                )
            )
        )

    def test_extract(self) -> None:
        image = Image.open(
            Path(__file__).parent / "data" / "images" / "monty_python.jpg"
        )
        boxes_and_patches = list(self.detector.extract(image))
        boxes_and_probabilities = list(self.detector.detect(image))
        self.assertSetEqual(
            {box for box, _ in boxes_and_patches},
            {box for box, _ in boxes_and_probabilities},
        )
        self.assertEqual(len(boxes_and_patches), 8)
        patches = list(patch.detach().cpu().numpy() for _, patch in boxes_and_patches)
        self.assertTrue(all(patch.shape == (3, 160, 160) for patch in patches))

        for query in (
            "eric-idle.npy",
            "graham-chapman.npy",
            "john-cleese.npy",
            "michael-palin.npy",
            "terry-gilliam.npy",
            "terry-jones.npy",
        ):
            self.assertTrue(
                any(
                    np.array_equal(
                        np.load(Path(__file__).parent / "data" / "patches" / query),
                        patch,
                    )
                    for patch in patches
                )
            )

        # image w/o faces
        self.assertFalse(
            list(
                self.detector.extract(
                    Image.open(Path(__file__).parent / "data" / "images" / "cactus.jpg")
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
