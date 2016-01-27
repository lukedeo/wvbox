from setuptools import setup
from setuptools import find_packages

try:
    from spacy.en import English
except ImportError:
    import logging
    logging.warn('spaCy (https://spacy.io) not found. Continue at your own risk.')

setup(name='WVBox',
      version='0.0.1',
      description='Library for boxing up word vector operations.',
      author='Luke de Oliveira',
      author_email='lukedeo@ldo.io',
      url='https://github.com/lukedeo/wvbox',
      packages=find_packages())
