from setuptools import setup, find_packages

setup(
    name="x-unet",
    packages=find_packages(exclude=[]),
    version="0.1.0",
    license="MIT",
    description="Unetlib",
    long_description_content_type="text/markdown",
    author="Krish Ghosh",
    author_email="krishparadox@gmail.com",
    url="https://github.com/krishparadox/unetlib",
    keywords=[
        "artificial intelligence",
        "deep learning",
        "stable diffusion",
        "segmentation",
        "unets",
    ],
    install_requires=[
        "einops>=0.4",
        "torch>=1.6",
    ],
    classifiers=[
        "Development Status :: 1 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
)
