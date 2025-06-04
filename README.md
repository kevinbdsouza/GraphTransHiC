# GraphTransHiC

GraphTransHiC implements a Graph Transformer model for analyzing Hi-C data.
It is built on top of the architecture described in [README_2.md](README_2.md).

## Setup
Create the environment with:
```bash
conda env create -f environment_gpu.yml
```

## Usage
Datasets are stored in `data/HiC`. Train or evaluate the model with:
```bash
python main_molecules_graph_regression.py
```

See `README_2.md` for more details on the underlying Graph Transformer.

