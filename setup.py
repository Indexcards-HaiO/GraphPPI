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
    packages=["graphppi", "graphppi.models", "graphppi.baselines"],
    package_dir={"graphppi": "src"},
    entry_points={
        "console_scripts": [
            "graphppi=graphppi.cli:main",
        ],
    },
)

