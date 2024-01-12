
# Faces
An educational face detection and identification module

## Introduction

This is a python module to detect faces in an image, and to identify each of them.

Face identification works in two steps.
First, we detect faces in the image.
At this stage, we don't care about who's face we find, the goal is to find all faces in the image.
Second, we identify a previously found face, i.e. we want to show the name of the depicted person.
To be able to do this,
we need a database that contains people's names and some example images of their face.
The idea is that we can compare the query face to all example images from that database
and use the name associated with the closest match.

The module is intended as an educational demonstration of face identification.
The heavy lifting is done by FaceNet [^1].
Specifically, this module is built on the [pytorch](https://pytorch.org) implementation of FaceNet [^2].
FaceNet comes with two components.
The first component can detect faces in an image (i.e., the first step of our pipeline).
This component (MTCNN) was trained on millions of images to detect and locate faces.

But simply comparing images of faces to each other (i.e., the second step of our pipeline)
would not give us good results.
The reason is that images are taken at different angles or light conditions,
and the person's appearance might differ across images due to hair styles or worn accessories.
Instead, we need to convert the images into a format that is independent of such factors.
That format should only express the core characteristics of a face, like the eye color,
face geometry (nose width, jawline, eye distance), etc.
We call this an encoding.
The second component FaceNet does just that.
It uses another deep learning model (Inception-ResNet) which was again trained on a huge
number of faces to carve out the "essence" of a face.

This module uses FaceNet to detect, extract, and compare images,
and offers the convenience and tooling around the bare FaceNet components.
For example, you can detect and identify faces with a few simple commands,
or you can tune the sensitivity of these components through the accompanying notebooks
to get better performance on your specific image library.


## Installation

Set up a virtual environment:

```bash
virtualenv env
source env/bin/activate
```

**NOTE**: On Raspberry Pi, install numpy and pillow before continuing:
```bash
pip install numpy==1.26.2 pillow==10.1.0
```

**NOTE**: install torch version 2.0.1 according to https://pytorch.org before continuing!
If you don't have a GPU, i.e., want to run everything on the **CPU**, use
```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
```

If you have a **GPU**, use
```bash
pip install torch==2.0.1 torchvision==0.15.2
```


Install faces as editable from the git repository:

```bash
git clone https://github.com/igsor/faces
cd faces
pip install -e .
```

If you want to tune the detection and identification sensitivity in the notebooks, install faces with the respective extras:

```bash
pip install -e ".[notebook]"
```

**NOTE**: (optional) install ipython kernel with the following command

```bash
python -m ipykernel install --user --name faces --display-name "Faces"
```

If you want to develop (*dev*), install faces with the respective extras:

```bash
pip install -e ".[dev]"
```

To ensure code style discipline, run the following commands:

```bash
isort faces
black faces
coverage run ; coverage html ; xdg-open .htmlcov/index.html
pylint faces
mypy
```

To build the package, do:

```bash
python -m build
```

To run only the tests (without coverage), run the following command from the **test folder**:

```bash
python -m unittest
```

To build the documentation, run the following commands from the **docs folder**:

```bash
sphinx-apidoc -f -o source/api ../faces/ --module-first -d 1 --separate
make html
xdg-open build/html/index.html
```


## Face detection

This module prepares the image, calls the necessary FaceNet functions,
and returns the result in nicely wrapped data structures.
See [detector.py](https://github.com/igsor/faces/blob/main/faces/detector.py) for details.

You can use this functionality in just one simple command:
```bash
faces detect data/douglas_adams.jpg
```

Or, you can use the following template to do the same in python code:
```python
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
builder.annotate(image, (box for box, _ in builder.detector.detect(image))).show()
```

You may have wondered why you can just execute the face detection without training.
This is because FaceNet has already been trained on millions of images,
so you don't need to do any training on your own dataset anymore.
That's actually great, because building and maintaining such a large dataset is
an elaborate and time-consuming effort,
and the training itself also takes a long time even on powerful hardware.

However, you can still tune its sensitivity to your own dataset.
Despite being quite accurate, FaceNet still hallucinates faces in an image,
i.e., it will present you locations where it believes to have found a face,
even though you can clearly see that there is none.

For each (potential) face that FaceNet detects, it will also specify how confident it is.
By rejecting faces with a low confidence you will get fewer such hallucinations (false positives).
Where exactly to set this confidence cut-off cannot be answered generally,
because a higher threshold (i.e., require high confidence) will also
make it miss more actual faces (false negatives).
This is a natural sensitivity trade-off:
High sensitivity leads to many face detections, including many wrong ones.
Low sensitivity leads to few detections but also few wrong ones.
Which threshold is best depends on your specific scenario.

Check out the [face detection tuning notebook](https://github.com/igsor/faces/blob/main/notebooks/detect.ipynb) for more details and instructions on how to tune the sensitivity to your dataset.


## Face identification

Face identification builds on top of face detection.
We again use FaceNet [^1] in this stage, but have to build some more logic around
it to make it usable.
Specifically, you'll need to build a database that contains reference face images
and who they're depicting.

Use the following command to do so for the an example image.
Note that you'll need to run this command no matter whether you continue with
the command-line or python shell.
```bash
faces db add data/douglas_adams.jpg
```

This command detects Douglas Adams' face and adds it to the database (stored in `~/.faces.pkl` by default).
You don't have to supply the name, because we infer it from the filename in this case
(run `import faces.main ; help(faces.main.Main.register)`
in a python shell to find out how this process works exactly).

You can repeat the command on more images to add more people to your database.
In fact, you must do so for all people that you want to be able to identify.
You can (and should, if possible) do this repeatedly with multiple images of the same person,
so that you have multiple reference images for one person,
which increases the likelihood that you will successfully identify them!

From now on, you can identify Douglas Adams in images.
Try this on the command-line:
```bash
faces identify data/who-is-this.jpg
```

Alternatively, you can achieve the same with some lines of python code:
```python
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
```

Same as with face detection,
you can also tune the sensitivity of the face identification.
An identifier with low sensitivity will only select references from the database
if they are extremely similar to your query image.
Conversely, it will reject any image for which you don't have a good enough reference
(it will label the face with "Anonymous" instead of the actual name),
even though you might have the person in your database.
A highly sensitive identifier will try hard to find a reference in the database.
But it will sometimes try too hard and match references of your database that depict
a completely different person than the one in your query image.

So, you're again faced with this trade-off,
and only you can decide what the correct cut-off value is.
To do so, have a look at the [face identification tuning notebook](https://github.com/igsor/faces/blob/main/notebooks/identify.ipynb).


## References

[^1]: F. Schroff, D. Kalenichenko, J. Philbin. FaceNet: A Unified Embedding for Face Recognition and Clustering, arXiv:1503.03832, 2015. [PDF](https://arxiv.org/pdf/1503.03832.pdf)
[^2]: [https://github.com/timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch)

