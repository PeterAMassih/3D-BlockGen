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
        "objaverse",
        "pytorch3d",  # Added for visualization
        "pillow",  # Added for image processing
        "imageio",  # Added for GIF creation
        "iopath",  # Added for PyTorch3D
    ],
    python_requires=">=3.10",
)
