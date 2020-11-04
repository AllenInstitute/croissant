from setuptools import setup, find_packages
import os.path
import codecs

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            # __version__ = "0.9"
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setup(
      name='croissant-ml',
      version=get_version("croissant/__init__.py"),
      description=("Classification of neurons segmented from two photon"
                   "microscopy videos"),
      author="Kat Schelonka, Isaak Willett, Dan Kapner, Nicholas Mei",
      author_email='kat.schelonka@alleninstitute.org',
      url="https://github.com/AllenInstitute/croissant",
      packages=find_packages(),
      setup_requires=['setuptools_scm'],
      install_requires=required,
      python_requires='>=3.7.7',
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Science/Research",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8"
      ]
)
