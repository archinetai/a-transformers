from setuptools import find_packages, setup

setup(
    name="a-transformers",
    packages=find_packages(exclude=[]),
    version="0.0.1",
    license="MIT",
    description="A-Transformers",
    long_description_content_type="text/markdown",
    author="Flavio Schneider",
    author_email="archinetai@protonmail.com",
    url="https://github.com/archinetai/a-transformers",
    keywords=["artificial intelligence", "deep learning"],
    install_requires=[
        "torch>=1.6",
        "data-science-types>=0.2",
        "einops>=0.4",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)