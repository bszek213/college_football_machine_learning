#from setuptools import setup, find_packages
from setuptools import setup
setup(
   name='cfb_ml',
   version='0.0.1',
   author='Brian Szekely',
   author_email='bszekely@nevada.unr.edu',
   packages=['cfb_ml', 'cfb_ml.test'],
   scripts=['src/vedb_odometry/vedb_calibration.py'],
   #url='http://pypi.python.org/pypi/PackageName/',
   license='LICENSE.txt',
   description='College football machine learning and data extraction',
   long_description=open('README.md').read(),
   install_requires=[
   "numpy >= 1.22.2",
   #"rigid-body-motion >= 0.9.1",# Not pip installable, must be installed from conda or git
   #"pupil_recording_interface >= 0.5.0", # Not pip installable, must be installed from conda or git
   "xarray >= 0.21.1",
   "scipy >= 1.8.0",
   #"vedb-store >= 0.0.1", # Not pip installable, must be installed from git
   "plotly >= 5.6.0",
   "pandas >= 1.4.1",
   "matplotlib==3.5.2"
   ],
)
