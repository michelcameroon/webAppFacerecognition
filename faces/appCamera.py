
# import the opencv library
import cv2
from datetime import datetime
# import the faces library
from faces.builder import DefaultBuilder
from faces.types import Image

# create a builder
builder = DefaultBuilder.from_defaults()

# define a video capture object
vid = cv2.VideoCapture(0)

while(True):


    # Capture the video frame
    ret, raw_image = vid.read()
    image = Image.from_array(raw_image)

    # identify faces in the image
    extracts = [
        (bounding_box, face_patch, builder.identifier(face_patch))
        for bounding_box, face_patch in builder.detector.extract(image)
    ]

    # annotate the image show it
    builder.annotate.with_identity(
        image, ((bbox, identity)
        for bbox, _, identity in extracts
    )).save("static/faceCapture.jpg")

    # show status
    print("captured image at", datetime.now().isoformat())

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

