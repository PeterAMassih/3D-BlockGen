from setuptools import setup, find_packages

setup(
    name="blockgen",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "matplotlib",
        "numpy",
        "tqdm",
        "diffusers",
        "transformers",
        "accelerate",
        "scipy",
        "scikit-learn",
        "pandas",
        "trimesh",
        "objaverse"
    ],
    python_requires=">=3.10",  # Specifically for Python 3.10+
)