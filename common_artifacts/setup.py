import os

from setuptools import find_packages, setup

"""
Взято отсюда
https://github.com/catalyst-team/codestyle/blob/master/setup.py
"""

# Package meta-data.
NAME = "heart_desease_common"
VERSION = "0.2.0"
DESCRIPTION = "Common.Entities"
URL = "http//github.com/made-ml-in-prod-2021/exotol@homework1"
AUTHOR = "exotol"
REQUIRES_PYTHON = ">=3.7.0"
LICENSE = "GPL"

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def load_requirements(filename):
    """Load package requirements."""
    with open(os.path.join(PROJECT_ROOT, filename), "r") as f:
        return f.read().splitlines()


extras = {
    "tests": load_requirements("requirements/requirements-tests.txt")
}

setup(
    name=NAME,
    packages=find_packages(exclude=("tests",)),
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    license=LICENSE,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    download_url=URL,
    install_requires=load_requirements("requirements/requirements.txt"),
    extras_require=extras
)
