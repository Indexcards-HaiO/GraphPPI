# GraphPPI Docker Image
# Build:  docker build -t graphppi .
# Run:    docker run --rm -it graphppi evaluate --k 3
# Mount:  docker run --rm -it -v $(pwd)/results:/app/results graphppi rank

FROM continuumio/miniconda3:24.11.1-0

LABEL org.opencontainers.image.title="GraphPPI"
LABEL org.opencontainers.image.description="GNN for PPI Link Prediction"
LABEL org.opencontainers.image.source="https://github.com/Totheinfinity66y/GraphPPI_test"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy environment file and create conda env
COPY environment.yml /app/environment.yml
RUN conda env create -f environment.yml && conda clean -afy

# Copy source code
COPY . /app

# Install graphppi in the conda environment
RUN /bin/bash -c "source activate graphppi && pip install -e . --no-deps"

# Make conda environment available by default
SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/envs/graphppi/bin:$PATH

# Default command
ENTRYPOINT ["graphppi"]
CMD ["--help"]
