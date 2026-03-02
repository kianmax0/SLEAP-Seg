from setuptools import setup, find_packages

setup(
    name="sleap-seg",
    version="0.1.0",
    description="Segmentation-guided pose estimation pipeline for multi-animal tracking",
    author="",
    python_requires=">=3.10",
    packages=find_packages(exclude=["scripts", "tests"]),
    entry_points={
        "console_scripts": [
            "sleap-seg=cli.run:main",
        ],
    },
    install_requires=[
        "ultralytics>=8.0",
        "segment-anything",
        "sleap",
        "filterpy>=1.4",
        "opencv-python>=4.7",
        "click>=8.1",
        "pyyaml>=6.0",
        "tqdm>=4.65",
        "numpy>=1.23",
        "scipy>=1.9",
        "h5py>=3.7",
        "lap>=0.4",
    ],
)
