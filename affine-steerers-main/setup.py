from setuptools import setup, find_packages
import os

setup(
    name="affine_steerers",
    packages=find_packages(include=[
        "affine_steerers*"
    ]),
    install_requires=[
        
    ],
    python_requires='>=3.9.0',
    version="0.0.1",
    author="Georg Bökman",
)
