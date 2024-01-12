import typing

from PIL import Image

EXIF_ORIENTATION_KEY = 274


def preprocess(
    img: Image.Image,
    target_size: int = 1000,
    rotate: typing.Optional[int] = None,
) -> Image.Image:
    """Preprocess an image.

    1. Scale larger side to *target_size*
    2. Rotate by angle *rotate*, or auto-rotate if *rotate=None* (the default).

    """
    # scale image
    if img.size[0] > img.size[1]:  # landscape
        img = img.resize(
            (target_size, int(img.height / img.width * target_size)), reducing_gap=3
        )
    elif img.size[0] < img.size[1]:  # portrait
        img = img.resize(
            (int(img.width / img.height * target_size), target_size), reducing_gap=3
        )

    # rotate image (if need be)
    if rotate is None:
        # auto-rotate according to EXIF information
        img_ori = img.getexif().get(EXIF_ORIENTATION_KEY, None)
        if img_ori == 3:
            img = img.rotate(180, expand=True)
        elif img_ori == 6:
            img = img.rotate(270, expand=True)
        elif img_ori == 8:
            img = img.rotate(90, expand=True)
    elif rotate != 0:
        # manually rotate
        img = img.rotate(rotate, expand=True)

    return img
