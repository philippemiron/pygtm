from setuptools import setup

setup(name='pygtm',
      version='0.1',
      description='python Geospatial Transition Matrix (pygtm) toolsets',
      url='http://github.com/philippemiron/pygtm',
      author='Philippe Miron',
      author_email='pmiron@miami.edu',
      license='MIT',
      packages=['pygtm'],
      zip_safe=False, install_requires=['numpy', 'scipy', 'cartopy', 'scikit-learn', 'matplotlib', 'netCDF4'])