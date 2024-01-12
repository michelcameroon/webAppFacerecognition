import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Set, Tuple

import torch

from faces import FacePatch, Identity, Registry


class InMemoryRegistry(Registry):
    """Store faces in volatile memory."""

    data: Set[Tuple[FacePatch, Identity]]

    def __init__(self):
        self.data = set()

    def add(self, face_patch: FacePatch, identity: Identity) -> None:
        self.data.add((face_patch, identity))

    def remove(self, identity: Identity) -> None:
        self.data = {
            (face_patch, id_) for face_patch, id_ in self.data if id_ != identity
        }

    def __iter__(self) -> Iterator[Tuple[FacePatch, Identity]]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class PickleRegistry(Registry):
    """Store faces and identities via pickle."""

    path: Path

    data: Set[Tuple[FacePatch, Identity]]

    @classmethod
    def open(cls, path: Path, device: torch.device) -> Registry:
        """Open the registry at *path*."""
        if not path.exists():
            return cls(path=path, data=set())
        with open(path, "rb") as registry_file:
            return cls(
                path=path,
                data={
                    (patch.to(device), identity)
                    for patch, identity in pickle.load(registry_file)["data"]
                },
            )

    def _save(self) -> None:
        with open(self.path, "wb") as registry_file:
            pickle.dump(
                {
                    "data": self.data,
                },
                registry_file,
            )

    def remove(self, identity: Identity) -> None:
        self.data = {
            (face_patch, id_) for face_patch, id_ in self.data if id_ != identity
        }
        self._save()

    def add(self, face_patch: FacePatch, identity: Identity) -> None:
        # NOTE: tensor hashes differ even if they are have identical values
        if knows_patch_as := {
            identity for patch, identity in self.data if torch.equal(face_patch, patch)
        }:
            if knows_patch_as != {identity}:
                raise ValueError(f"already known as {knows_patch_as}")
            return

        self.data.add((face_patch, identity))
        self._save()

    def __iter__(self) -> Iterator[Tuple[FacePatch, Identity]]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)
