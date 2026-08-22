from setuptools import setup, find_packages

setup(
    name="gdn2-pallas",
    version="0.1.0",
    description="Gated DeltaNet-2 with fused Pallas kernels, optimized for TPU v5e",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Akseleu Omirbay",
    author_email="your-email@example.com",   # замени на реальный
    url="https://github.com/Akseleu-J/gdn2-pallas",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "jax>=0.4.20",
        "jaxlib>=0.4.20",
        "numpy>=1.24",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
)
