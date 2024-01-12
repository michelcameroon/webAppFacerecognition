from typing import Iterable, Iterator, Tuple

import numpy as np
import torch
from facenet_pytorch import MTCNN

from faces import BoundingBox, Detector, FacePatch, FaceProbability, Image


class MTCNNDetector(Detector):
    """Use the MTCNN network to detect and extract faces."""

    probability_threshold: float

    model: MTCNN

    device: torch.device

    def __init__(
        self,
        # torch device.
        device: torch.device,
        # minimum face probability. Smaller probability will detect more possible faces.
        probability_threshold: float = 0.9,
        # minimum face size. Smaller values will detect more possible faces.
        min_face_size: int = 20,
        # internal probability thresholds. Decrease values to increase the sensitivity.
        thresholds: Tuple[float, float, float] = (0.6, 0.7, 0.7),
        # pyramid scaling factor.
        factor: float = 0.709,
        # size of the extracted patch.
        patch_size: int = 160,
    ):
        self.device = device
        self.probability_threshold = probability_threshold
        # initialize the face detection network
        self.model = MTCNN(
            min_face_size=min_face_size,
            thresholds=thresholds,
            factor=factor,
            device=device,
            keep_all=True,
            image_size=patch_size,
        )

    def detect(self, image: Image) -> Iterable[Tuple[BoundingBox, FaceProbability]]:
        boxes, probs = self.model.detect(image.image)
        if boxes is None:  # no boxes to return
            return
        for box, prob in zip(boxes, probs):
            if prob >= self.probability_threshold:
                yield BoundingBox(*box), prob

    def extract(self, image: Image) -> Iterator[Tuple[BoundingBox, FacePatch]]:
        for box, _ in self.detect(image):
            yield box, self.model.extract(
                image.image, np.array(box.as_tuple).reshape(1, -1), None
            ).squeeze(0).to(self.device)
