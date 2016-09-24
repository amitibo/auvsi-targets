#!/usr/bin/env python

"""
AUVSItargets: Training samples generation for the ADLC task in the AUVSI SUAS
competition.

Authors: See AUTHORS file.
License: See LICENSE file
"""

from setuptools import setup, find_packages
import os

VERSION = "1.0"


def main():
    """main setup function"""

    setup (
        name = "AUVSItargets",
        version = VERSION,
        description="Training samples generation",
        long_description="Training samples generation for the ADLC task in the AUVSI SUAS competition.",
        author="Amit Aides, Ahmad Kiswani",
        author_email="", # Removed to limit spam harvesting.
        url="https://github.com/amitibo/AUVSItargets/",
        packages = find_packages(),
        license="BSD",
        zip_safe=False
    )


if __name__ == '__main__':
    main()
