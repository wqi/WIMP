#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='What-If Motion Prediction',
    version='0.1.0',
    description='Official reference code for the What-If Motion Prediction paper.',
    author='William Qi, Siddhesh Khandelwal',
    author_email='wq@cs.cmu.edu, skhandel@cs.ubc.ca',
    url='https://github.com/wqi/WIMP',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)
