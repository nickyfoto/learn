"""Minimal setup file for learn project."""

from setuptools import setup, find_packages

setup(
    name='learn',
    version='0.1.0',
    description='Learn Fundamental Machine Learning Algorithms',

    author='Qiang Huang',
    author_email='nickyfoto@gmail.com',
    url='https://macworks.io',

    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)
