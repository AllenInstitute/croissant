from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setup(
      name='croissant',
      use_scm_version=True,
      description=("Classification of neurons segmented from two photon"
                   "microscopy videos"),
      author="Kat Schelonka, Isaak Willett, Dan Kapner, Nicholas Mei",
      author_email='kat.schelonka@alleninstitute.org',
      url="https://github.com/AllenInstitute/croissant",
      packages=find_packages(),
      setup_requires=['setuptools_scm'],
      install_requires=required,
)
