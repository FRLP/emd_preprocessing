from setuptools import setup, find_packages

setup(
    name="emd_preprocessing",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "clarabel==0.9.0",
        "numpy>=1.21",
        "scipy>=1.7",
        "cvxpy>=1.1",
        "matplotlib>=3.5",
        "opencv-python>=4.5",
        "hnswlib>=0.7",
        "tqdm>=4.64",
        "spacy>=3.0",
        "mecab-python3>=1.0",
        "ipadic>=1.0",
        "sympy>=1.14",
        "gensim>=4.3",
        "unidic-lite>=1.0"
    ],
)