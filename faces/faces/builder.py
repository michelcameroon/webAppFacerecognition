from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Tuple

import torch

from faces import Annotate, Builder, Detector, Encoder, Identifier, Identity, Registry
from faces.detector import MTCNNDetector
from faces.drawing import PILAnnotate
from faces.encoder import ResnetEncoder
from faces.identifier import ConstrainedNearestNeighbourClassifier
from faces.registry import PickleRegistry


# pylint: disable=too-many-instance-attributes
@dataclass
class DefaultBuilder(Builder):
    """Build classes from default arguments."""

    device: torch.device

    registry_path: Path

    probability_threshold: float = 0.9

    distance_threshold: float = 1.0

    restklasse: Identity = "Anonymous"

    min_face_size: int = 20

    thresholds: Tuple[float, float, float] = (0.6, 0.7, 0.7)

    factor: float = 0.709

    @cached_property
    def annotate(self) -> Annotate:
        return PILAnnotate()

    @cached_property
    def identifier(self) -> Identifier:
        return ConstrainedNearestNeighbourClassifier.fit(
            samples=self.registry,
            distance_threshold=self.distance_threshold,
            restklasse=self.restklasse,
            encoder=self.encoder,
        )

    @cached_property
    def encoder(self) -> Encoder:
        return ResnetEncoder(
            device=self.device,
        )

    @cached_property
    def detector(self) -> Detector:
        return MTCNNDetector(
            device=self.device,
            probability_threshold=self.probability_threshold,
            min_face_size=self.min_face_size,
            thresholds=self.thresholds,
            factor=self.factor,
        )

    @property
    def registry(self) -> Registry:
        return PickleRegistry.open(self.registry_path, self.device)

    @classmethod
    def from_args(cls, args) -> Builder:
        device = args.device
        if not device:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        return cls(
            device=torch.device(device),
            registry_path=args.registry_path,
            probability_threshold=args.probability_threshold,
            distance_threshold=args.distance_threshold,
        )

    @classmethod
    def from_defaults(cls) -> Builder:
        return cls(
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            registry_path=Path("~/.faces.pkl").expanduser(),
        )
