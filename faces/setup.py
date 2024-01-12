
from setuptools import setup, find_packages
import os

setup(
    # package metadata
    name='faces',
    version='0.23.08',
    author='Matthias Baumgartner',
    author_email='dev@igsor.net',
    description='Face detection, extraction, and identification. An example.',
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    license='BSD',
    license_files=('LICENSE', ),
    url='https://www.igsor.net/faces/',

    # packages
    packages=find_packages(include=['faces']),
    package_dir={'faces': 'faces'},
    # data files are included if mentioned in MANIFEST.in
    include_package_data=True,

    # entrypoints
    entry_points={
        'console_scripts': [
            'faces= faces.main:main',
            ],
        },

    # dependencies
    python_requires=">=3.7",
    install_requires=(
        'matplotlib==3.8.2',
        'numpy==1.26.2',
        'opencv-python==4.8.1.78',
        'pillow==10.1.0',
        'torch==2.0.1',
        'facenet_pytorch==2.5.3',
        ),
    extras_require={
        'dev': [
            'black',
            'build',
            'coverage',
            'furo',
            'isort',
            'mypy',
            'pylint',
            'sphinx<7.2.0',
            ],
        'notebook': [
            'ipykernel',
            'ipywidgets',
            'pandas',
            'plotly-express',
            'scikit-learn',
            'tqdm',
            'nbformat>=4.2.0',
            ],
        },
)
