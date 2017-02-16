#!/usr/bin/env python
from setuptools import setup

setup(
    name='picross',
    version='0.0.1',
    author='CJ Carey',
    author_email='perimosocordiae@gmail.com',
    description='Picross puzzle solver',
    url='http://github.com/perimosocordiae/picross',
    license='MIT',
    packages=['picross'],
    install_requires=[
        'numpy',
        'webtool',
        'six',
    ],
    scripts=['solve_picross.py'],
)
