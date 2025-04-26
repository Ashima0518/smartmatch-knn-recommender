from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="smartmatch",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A KNN-based movie recommendation engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/smartmatch-knn-recommender",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
    ],
) 