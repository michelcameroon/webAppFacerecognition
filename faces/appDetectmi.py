from flask import Flask

import faces

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello, World!'

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
    Main().detect(builder, image).show()

    # or do the same by calling the detector directly
    imgmi = builder.annotate(image, (box for box, _ in builder.detector.detect(image)))
    


    return 'identmi'
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
    Main().identify(builder, image).show()

    # or do the same by calling the detector and identifier directly
    builder.annotate.with_identity(image, (
        (box, builder.identifier(patch))
        for box, patch in builder.detector.extract(image))).show()


    return 'identmi'

