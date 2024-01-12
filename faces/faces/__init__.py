from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from functools import cached_property
from pathlib import Path
from typing import Any, Tuple

import torch
from PIL import Image as PILImage

from faces.types import (
    BoundingBox,
    FaceEncoding,
    FacePatch,
    FaceProbability,
    Identity,
    Image,
    VideoFrame,
)


class Identifier(ABC):
    """Identify faces."""

    @abstractmethod
    def __call__(self, face_patch: FacePatch) -> Identity:
        """Return the identity of the person in *face_patch*."""


class Detector(ABC):
    """Detect faces."""

    @abstractmethod
    def detect(self, image: Image) -> Iterable[Tuple[BoundingBox, FaceProbability]]:
        """Return the bounding boxes and likelihoods of there being a face."""

    @abstractmethod
    def extract(self, image: Image) -> Iterable[Tuple[BoundingBox, FacePatch]]:
        """Return the bounding boxes and faces detected in an image."""


class Encoder(ABC):
    """Encode a face patch."""

    @abstractmethod
    def __call__(self, face_patch: FacePatch) -> FaceEncoding:
        """Return the encoding of a *face_path*."""

    @abstractmethod
    def many(self, patches: torch.Tensor) -> torch.Tensor:
        """Return N encodings of face *patches* given as an (N, ...) tensor."""


class Registry(ABC):
    """Face patches and identities storage."""

    @abstractmethod
    def add(self, face_patch: FacePatch, identity: Identity) -> None:
        """Store a face and its identity. Auto-commits."""

    @abstractmethod
    def remove(self, identity: Identity) -> None:
        """Remove an identity and all its faces. Auto-commits."""

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[FacePatch, Identity]]:
        """Iterate over face patches and their identities."""


class Annotate(ABC):
    """Annotate an image with bounding boxes and additional information."""

    @abstractmethod
    def with_probability(
        self,
        image: Image,
        boxes_and_probability: Iterable[Tuple[BoundingBox, FaceProbability]],
    ) -> PILImage.Image:
        """Draw bounding boxes and their likelihood of enclosing a face."""

    @abstractmethod
    def with_identity(
        self,
        image: Image,
        boxes_and_identity: Iterable[Tuple[BoundingBox, Identity]],
    ) -> PILImage.Image:
        """Draw bounding boxes and their identity."""

    @abstractmethod
    def with_enumeration(
        self, image: Image, boxes: Iterable[BoundingBox], start: int = 0
    ) -> PILImage.Image:
        """Draw bounding boxes and their index in the sequence."""

    @abstractmethod
    def __call__(self, image: Image, boxes: Iterable[BoundingBox]) -> PILImage.Image:
        """Draw bounding boxes."""


class Builder(ABC):
    """Build instances."""

    @cached_property
    @abstractmethod
    def annotate(self) -> Annotate:
        """Return an Annotate instance."""

    @cached_property
    @abstractmethod
    def identifier(self) -> Identifier:
        """Return a Identifier instance."""

    def reload(self) -> Builder:
        """Reload the builder's persistent parts from disc."""
        del self.identifier
        return self

    @cached_property
    @abstractmethod
    def encoder(self) -> Encoder:
        """Return an Encoder instance."""

    @property
    @abstractmethod
    def registry(self) -> Registry:
        """Return a Registry instance."""

    @cached_property
    @abstractmethod
    def detector(self) -> Detector:
        """Return a Detector instance."""

    @classmethod
    @abstractmethod
    def from_args(cls, args: argparse.Namespace) -> Builder:
        """Initialize the Builder from argparse arguments."""

    @classmethod
    @abstractmethod
    def from_defaults(cls) -> Builder:
        """Initialize the Builder from default arguments."""
