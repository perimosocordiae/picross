#!/usr/bin/env python
from setuptools import setup

setup(
    name='picross',
    version='0.1.0',
    author='CJ Carey',
    author_email='perimosocordiae@gmail.com',
    description='Picross puzzle solver',
    url='http://github.com/perimosocordiae/picross',
    license='MIT',
    packages=['picross'],
    install_requires=[
        'numpy >= 1.11',
        'scipy',
        'webtool',
        'six',
        'matplotlib >= 2.0',
        'scikit-learn',
        'python-opencv >= 3.2',
    ],
    scripts=['solve_picross.py'],
)
