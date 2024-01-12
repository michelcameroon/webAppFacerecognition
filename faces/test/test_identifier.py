import unittest
from os.path import basename
from pathlib import Path

import numpy as np
import torch

from faces import FacePatch, Identity
from faces.encoder import ResnetEncoder
from faces.identifier import ConstrainedNearestNeighbourClassifier


class TestIdentifier(unittest.TestCase):
    def setUp(self) -> None:
        self.encoder = ResnetEncoder(torch.device("cpu"))

    def test_fit(self) -> None:
        samples = [
            (
                FacePatch(np.load(Path(__file__).parent / "data" / "patches" / path)),
                Identity(basename(path)),
            )
            for path in (
                "eric-idle.npy",
                "graham-chapman.npy",
                "john-cleese.npy",
                "michael-palin.npy",
            )
        ]
        identifier = ConstrainedNearestNeighbourClassifier.fit(
            samples=samples,
            distance_threshold=1.1,
            restklasse="Anonymous",
            encoder=self.encoder,
        )
        self.assertEqual(identifier.classifier.encodings.shape, (4, 512))
        self.assertEqual(identifier.classifier.targets.shape, (4,))
        self.assertEqual(len(identifier.index2identity), 4)

        # empty data
        identifier = ConstrainedNearestNeighbourClassifier.fit(
            samples=[],
            distance_threshold=1.1,
            restklasse="Anonymous",
            encoder=self.encoder,
        )
        self.assertEqual(identifier.classifier.encodings.shape, (0,))
        self.assertEqual(identifier.classifier.targets.shape, (0,))
        self.assertEqual(len(identifier.index2identity), 0)

        # restklasse only
        identifier = ConstrainedNearestNeighbourClassifier.fit(
            samples=[(samples[0][0], "Anonymous")],
            distance_threshold=1.1,
            restklasse="Anonymous",
            encoder=self.encoder,
        )
        self.assertEqual(identifier.classifier.encodings.shape, (0,))
        self.assertEqual(identifier.classifier.targets.shape, (0,))
        self.assertEqual(len(identifier.index2identity), 0)

    def test_call(self) -> None:
        idle, chapman, *samples_train = [
            (
                FacePatch(np.load(Path(__file__).parent / "data" / "patches" / path)),
                Identity(basename(path)),
            )
            for path in (
                "eric-idle.npy",
                "graham-chapman.npy",
                "john-cleese.npy",
                "michael-palin.npy",
                "terry-gilliam.npy",
                "terry-jones.npy",
            )
        ]

        # non-empty identifier
        identifier = ConstrainedNearestNeighbourClassifier.fit(
            samples=samples_train,
            distance_threshold=1.1,
            restklasse="Anonymous",
            encoder=self.encoder,
        )
        for patch, target in samples_train:
            self.assertEqual(identifier(patch), target)

        self.assertEqual(identifier(idle[0]), "terry-jones.npy")
        self.assertEqual(identifier(chapman[0]), "Anonymous")

        # empty identifier
        identifier = ConstrainedNearestNeighbourClassifier.fit(
            samples=[],
            distance_threshold=1.1,
            restklasse="Anonymous",
            encoder=self.encoder,
        )
        self.assertEqual(identifier(idle[0]), "Anonymous")
        self.assertEqual(identifier(chapman[0]), "Anonymous")


if __name__ == "__main__":
    unittest.main()
