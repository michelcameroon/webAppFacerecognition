import torch
from facenet_pytorch import InceptionResnetV1

from faces import Encoder, FaceEncoding, FacePatch


class ResnetEncoder(Encoder):
    """Use InceptionResnet to encode face patches to a 512-dimensional embedding."""

    model: InceptionResnetV1

    def __init__(
        self,
        device: torch.device,
    ):
        self.model = InceptionResnetV1("vggface2", device=device).eval()

    def __call__(self, face_patch: FacePatch) -> FaceEncoding:
        # pylint: disable=not-callable
        return self.model(face_patch.unsqueeze(0)).squeeze(0)

    def many(self, patches: torch.Tensor) -> torch.Tensor:
        # pylint: disable=not-callable
        return self.model(patches)
