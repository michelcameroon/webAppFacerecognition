from dataclasses import dataclass
from typing import Iterable, Tuple

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from faces import Annotate, BoundingBox, FaceProbability, Identity, Image

FONT = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 30)


@dataclass(frozen=True)
class Color:
    """RGBA Color specification."""

    # color channels, in range [0, 255]
    red: int
    green: int
    blue: int
    # alpha channel, in range [0, 255]
    alpha: int = 0

    def __post_init__(self):
        assert 0 <= self.red <= 255
        assert 0 <= self.green <= 255
        assert 0 <= self.blue <= 255
        assert 0 <= self.alpha <= 255

    @property
    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return the color as (red, green, blue, alpha)-tuple."""
        return self.red, self.green, self.blue, self.alpha


@dataclass(frozen=True)
class PILAnnotate(Annotate):
    """Draw on images with PIL."""

    line_width: int = 6
    box_color: Color = Color(255, 0, 0)
    font_color: Color = Color(255, 255, 255, 0)

    def with_probability(
        self,
        image: Image,
        boxes_and_probability: Iterable[Tuple[BoundingBox, FaceProbability]],
    ) -> PILImage.Image:
        return self._annotate(
            image, ((box, f"{prob:0.5f}") for box, prob in boxes_and_probability)
        )

    def with_identity(
        self,
        image: Image,
        boxes_and_identity: Iterable[Tuple[BoundingBox, Identity]],
    ) -> PILImage.Image:
        return self._annotate(image, boxes_and_identity)

    def with_enumeration(
        self, image: Image, boxes: Iterable[BoundingBox], start: int = 0
    ) -> PILImage.Image:
        return self._annotate(
            image,
            ((box, str(index)) for index, box in enumerate(boxes, start)),
        )

    def __call__(self, image: Image, boxes: Iterable[BoundingBox]) -> PILImage.Image:
        return self._annotate(image, ((box, "") for box in boxes))

    def _annotate(
        self,
        image: Image,
        boxes_and_labels: Iterable[Tuple[BoundingBox, str]],
    ) -> PILImage.Image:
        """Draw bounding boxes and their labels into a copy of *img*. Return the new image.

        Parameters:

        * boxes: Bounding boxes. See `BoundingBox`.
        * line_width: Box line width.
        * show_text: Flag whether to show the box label or not.
        * box_color: Box line color. See `Color`.
        * font_color: Box label color. See `Color`.
        * label_format: Format string for box labels of float or int type.

        """
        image = image.image.copy()
        draw = ImageDraw.Draw(image)
        for box, label in boxes_and_labels:
            draw.rectangle(
                box.as_tuple,
                outline=self.box_color.as_tuple,
                width=self.line_width,
            )
            if label:
                position = box.lower_left, box.lower_top + self.line_width
                draw.text(
                    position,
                    text=label,
                    font=FONT,
                    fill=self.font_color.as_tuple,
                )

        return image
