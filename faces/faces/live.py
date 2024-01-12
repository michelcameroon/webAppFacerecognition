import logging
from datetime import datetime
from tempfile import mkstemp
from typing import Any

import cv2
import numpy as np

from faces import Builder, FacePatch, Identity, Image, VideoFrame

WINDOW_NAME = "continuous face identification"

from typing import Set


class Live:
    builder: Builder

    window_name: str

    capture: cv2.VideoCapture

    identified_in_session: Set[Identity]

    def __init__(
        self, builder: Builder, window_name: str = WINDOW_NAME, video_device: int = 0
    ):
        self.builder = builder
        self.window_name = window_name
        # initialize output window
        cv2.namedWindow(self.window_name)
        # initialize video capture
        self.capture = cv2.VideoCapture(video_device)
        # initialize session
        self.identified_in_session = set()

    def __del__(self):
        # cleanup
        self.capture.release()
        cv2.destroyWindow(self.window_name)

    def run(self):
        while True:
            # grab frame
            if not (video_frame := VideoFrame(*self.capture.read())).rval:
                break

            # load the image
            image = Image.from_array(video_frame.frame)

            # identify faces in the image
            extracts = [
                (bounding_box, face_patch, self.builder.identifier(face_patch))
                for bounding_box, face_patch in self.builder.detector.extract(image)
            ]

            # track identified people
            self.track_identified(
                {
                    identity
                    for _, _, identity in extracts
                    if identity != self.builder.identifier.restklasse
                }
            )

            # annotate the image show it
            cv2.imshow(
                self.window_name,
                np.array(
                    self.builder.annotate.with_identity(
                        image, ((bbox, identity) for bbox, _, identity in extracts)
                    )
                ),
            )

            if (key := cv2.waitKey(20)) == 27:  # ESC pressed
                return
            elif key == 32:  # SPACE pressed
                self.save_frame(image)
            elif key == 13:  # ENTER pressed
                try:
                    self.register_face(
                        {
                            patch
                            for _, patch, identity in extracts
                            if identity == self.builder.identifier.restklasse
                        }
                    )
                except ValueError as error:
                    logging.error(str(error))

    def track_identified(self, identified: Set[Identity]):
        """Handle identified faces."""
        for name in identified - self.identified_in_session:
            logging.info(f"found {name}")
        self.identified_in_session |= identified

    def save_frame(self, image: Image) -> None:
        """Save a frame to a file at an auto-generated path."""
        timestamp = datetime.now().isoformat()
        filename = mkstemp(prefix=f"faces-capture-{timestamp}-", suffix=".jpg")[1]
        image.image.save(filename)
        logging.info(f"captured image at {filename}")

    def register_face(self, unidentified: Set[FacePatch]):
        """Add a single unidentified face to the database."""
        if not unidentified:
            raise ValueError("requires one unidentified face to have been detected")
        if len(unidentified) != 1:
            raise ValueError("can only add one face at a time")

        # one unidentified face - ask user for identity
        while not (user_input := self._ask_for_identity()):
            pass

        try:
            (face_patch,) = unidentified
            self.builder.registry.add(face_patch, Identity(user_input))
            self.builder.reload()
        except ValueError as error:
            raise ValueError(f"skipping face: {error}") from error

    def _ask_for_identity(self) -> str:
        """Ask a user to identify a face."""
        try:
            print("Please specify the identity of the face.")
            print("Enter -1 to skip this face")
            if (answer := input(">>> ").strip()) == "-1":
                raise EOFError()
            return answer
        except EOFError as error:
            raise ValueError("skipping face") from error
