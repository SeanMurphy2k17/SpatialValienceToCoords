#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for Spatial Valence Solution package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="spatial-valence-solution",
    version="1.0.0",
    author="Sean",
    description="Enhanced spatial valence processor for AI consciousness systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/spatial-valence-solution",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
        ],
        "examples": [
            "matplotlib",
            "pandas",
        ],
    },
    keywords=[
        "spatial memory", 
        "semantic analysis", 
        "AI consciousness", 
        "natural language processing",
        "valence processing",
        "spatial coordinates"
    ],
    include_package_data=True,
    zip_safe=False,
) 