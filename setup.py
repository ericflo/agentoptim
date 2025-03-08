#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="agentoptim",
    version="2.0.0",
    author="Eric Florenzano",
    author_email="",
    description="MCP tools for data-driven prompt optimization and evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ericflo/agentoptim",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "agentoptim=agentoptim.__main__:main",
        ],
    },
    scripts=["bin/agentoptim"],
)