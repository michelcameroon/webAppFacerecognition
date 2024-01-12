from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Tuple

import torch

from faces import Encoder, FaceEncoding, FacePatch, Identifier, Identity


@dataclass(frozen=True)
class _NearestNeighbour:
    """Nearest neighbour classifier."""

    encodings: torch.Tensor

    targets: torch.Tensor

    def __post_init__(self) -> None:
        assert len(self.encodings) == len(self.targets)

    def __call__(self, encoding: FaceEncoding) -> Tuple[int, float]:
        """Return the nearest neighbour and its distance to *encoding*."""
        # pairwise distances
        dist = torch.cdist(encoding.unsqueeze(0), self.encodings).squeeze(0)
        # index of lowest distance
        min_distance, min_index = torch.min(dist, 0)
        # return identity and distance
        return int(self.targets[min_index].item()), min_distance.item()

    @classmethod
    def empty(cls) -> _NearestNeighbour:
        """Return a nearest neighbour classifier without references."""
        return cls(
            encodings=torch.empty((0,)),
            targets=torch.empty((0,)),
        )

    @property
    def is_empty(self) -> bool:
        """Return True if the classifier has no references."""
        return len(self.encodings) == 0


@dataclass(frozen=True)
class ConstrainedNearestNeighbourClassifier(Identifier):
    """Open-world nearest neighbour classifier.
    Uses a distance threshold on the nearest neighbour to
    discard samples that do not match any reference point.
    Compares patches through their encoding rather than
    directly.
    """

    encoder: Encoder

    distance_threshold: float

    restklasse: Identity

    index2identity: Mapping[int, Identity]

    classifier: _NearestNeighbour

    @classmethod
    def fit(
        cls,
        samples: Iterable[Tuple[FacePatch, Identity]],
        *,
        encoder: Encoder,
        distance_threshold: float = 1.0,
        restklasse: Identity = Identity("Anonymous"),
    ) -> Identifier:
        """Return an identifier that is fitted to *samples*."""
        # filter
        valid_samples = (
            (patch, label) for patch, label in samples if label != restklasse
        )
        try:
            # unpack
            patches, labels = zip(*valid_samples)
        except ValueError:
            # valid_samples was empty
            return cls(
                encoder=encoder,
                distance_threshold=distance_threshold,
                restklasse=restklasse,
                index2identity={},
                classifier=_NearestNeighbour.empty(),
            )

        # index/identity mappings
        index2identity = dict(enumerate(set(labels)))
        identity2index = {identity: index for index, identity in index2identity.items()}
        # classifier
        classifier = _NearestNeighbour(
            encodings=encoder.many(torch.stack(patches)),
            # NOTE: targets can be on the cpu no matter the encodings
            targets=torch.tensor(
                [identity2index[label] for label in labels], device=torch.device("cpu")
            ),
        )
        return cls(
            encoder=encoder,
            distance_threshold=distance_threshold,
            restklasse=restklasse,
            index2identity=index2identity,
            classifier=classifier,
        )

    def nearest_neighbour(self, face_patch: FacePatch) -> Tuple[Identity, float]:
        """Return the nearest neighbour and its distance."""
        if self.classifier.is_empty:
            return self.restklasse, float("inf")
        identity_index, distance = self.classifier(self.encoder(face_patch))
        return self.index2identity[identity_index], distance

    def __call__(self, face_patch: FacePatch) -> Identity:
        """Return the nearest neighbour's identity."""
        identity, dist = self.nearest_neighbour(face_patch)
        if dist > self.distance_threshold:
            return self.restklasse
        return identity
