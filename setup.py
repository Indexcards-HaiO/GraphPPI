from setuptools import setup

setup(
    name="graphppi",
    version="1.0.0",
    description="Graph Neural Network for PPI Link Prediction",
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torch_geometric>=2.5.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "networkx>=3.0",
        "node2vec>=0.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "docs": [
            "mkdocs>=1.5",
            "mkdocstrings[python]>=0.25",
            "mkdocs-material>=9.0",
        ],
    },
    packages=["graphppi", "graphppi.models", "graphppi.baselines"],
    package_dir={"graphppi": "src"},
    entry_points={
        "console_scripts": [
            "graphppi=graphppi.cli:main",
        ],
    },
)

