#!/usr/bin/env python3
from setuptools import find_packages, setup


setup(
    name="STOMP",
    version="0.1.0",
    author="",
    author_email="",
    description="A STOMP implementation with GridWorld module in Python",
    url="https://github.com/ramos-ai/STOMP",

    # Automatically find all packages
    packages=find_packages(),

    # Include additional files (if needed)
    include_package_data=True,

    # Python version requirement
    python_requires=">=3.7",

    # Dependencies
    install_requires=[
        "numpy==2.1.0",
        "matplotlib==3.9.4",
        "tqdm==4.66.5",
        "ipywidgets==8.1.6",
        "plotly==6.0.1",
        "nbformat==5.10.4",
        "ruff==0.11.8",
        "numba==0.61.2",
        "wandb==0.21.1",
    ],
)
