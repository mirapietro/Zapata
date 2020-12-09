import os
from setuptools import setup, find_packages


def read(*paths):
    """Build a file path from *paths* and return the contents."""
    with open(os.path.join(*paths), 'r') as f:
        return f.read()

setup(
    name = "Zapata",
    version = "1.0",
    author = "CMCC-Foundation",
    description = ("A revolutionary library for analysis and plotting of meteorological data"),
    license='GPLv3+',
    url = "https://github.com/CMCC-Foundation/Zapata",
    packages=['zapata','klus'],
    package_data={
        'zapata': ['*','../*.yml','../README.md','../LICENSE','../interp.py','../zeus.py'],
    },
    include_package_data=True,
    long_description=(read('README.md') + '\n\n'),
    classifiers=[
	'Development Status :: 4 - Beta',
	'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
	'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering',
    ],
)

