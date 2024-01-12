from flask import Flask, render_template


import faces

app = Flask(__name__)




@app.route('/')
def hello():
    return render_template('index.html')
#    return 'Hello, World!'

@app.route('/detectmi')
def detectmi():

    # import
    from pathlib import Path
    from faces import Image
    from faces.builder import DefaultBuilder
    from faces.main import Main

    # initialize pipeline
    builder = DefaultBuilder.from_defaults()

    # open an image
    image = Image.open(Path('data/douglas_adams.jpg'))

    # detect using the built-in function
    #Main().detect(builder, image).show()
    Main().detect(builder, image)

    # or do the same by calling the detector directly
    imgmi = builder.annotate(image, (box for box, _ in builder.detector.detect(image)))

    # save the image    
    #Main().detect(builder, image).save("images/face.jpg")
    Main().detect(builder, image).save("static/faceDetect.jpg")
    #Main().detect(builder, imgmi).save("images/face.jpg")

    return render_template('detectmi.html') 
    #return 'identmi'
    #return str(imgmi)
    #return imgmi


@app.route('/identmi')
def identmi():

    # import
    from pathlib import Path
    from faces import Image
    from faces.builder import DefaultBuilder
    from faces.main import Main

    # initialize pipeline
    builder = DefaultBuilder.from_defaults()

    # open an image
    image = Image.open(Path('data/who-is-this.jpg'))

    # identify using the built-in function
    #Main().identify(builder, image).show()
    Main().identify(builder, image).save("static/faceIdent.jpg")

    # or do the same by calling the detector and identifier directly
    '''
    builder.annotate.with_identity(image, (
        (box, builder.identifier(patch))
        for box, patch in builder.detector.extract(image)))
    '''
    #Main().detect(builder, image).save("static/faceIdent.jpg")

    
    #return 'identmi'
    return render_template('identmi.html') 



@app.route('/capturemi')
def capturemi():
    return render_template('capturemi.html') 

    '''

    import logging
    from datetime import datetime
    from tempfile import mkstemp
    from typing import Any

    import cv2
    import numpy as np

    import faces

    from faces import Builder, FacePatch, Identity, Image, VideoFrame
    #from faces import Builder, FacePatch, Identity, Image, VideoFrame, Live

    WINDOW_NAME = "continuous face identification"

    from typing import Set


    class Live:
        builder: Builder

        window_name: str

        capture: cv2.VideoCapture

        identified_in_session: Set[Identity]

        def __init__(self, builder: Builder, window_name: str = WINDOW_NAME, video_device: int = 0):
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
    
        def __init__(self, builder: Builder, window_name: str = WINDOW_NAME, video_device: int = 0):
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


    '''




if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)
