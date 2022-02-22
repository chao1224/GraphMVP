# Pre-training Molecular Graph Representation with 3D Geometry

**ICLR 2022**

Authors: Shengchao Liu, Hanchen Wang, Weiyang Liu, Joan Lasenby, Hongyu Guo, Jian Tang

[[Project Page](https://chao1224.github.io/GraphMVP)]
[[Paper](https://openreview.net/forum?id=xQUe1pOKPam)]
[[ArXiv](https://arxiv.org/abs/2110.07728)]<br>
[[NeurIPS SSL Workshop 2021](https://arxiv.org/abs/2110.07728)]

This repository provides the source code for the ICLR'22 paper **Pre-training Molecular Graph Representation with 3D Geometry**, with the following task:
- During pre-training, we consider both the 2D topology and 3D geometry.
- During downstream, we consider tasks with 2D topology only.

<p align="center">
  <img src="fig/pipeline.png" /> 
</p>

## Baselines
For implementation, this repository also provides the following graph SSL baselines:
- Generative Graph SSL:
  - [Edge Prediction (EdgePred)]()
  - [AttributeMasking (AttrMask)]()
  - [GPT-GNN]()
- Contrastive Graph SSL:
  - [InfoGraph]()
  - [Context Prediction (ContextPred)]()
  - [GraphLoG]()
  - [GraphCL]()
  - [JOAO]()
- Predictive Graph SSL:
  - [Grover]()

<p align="center">
  <img src="fig/baselines.png" /> 
</p>

## Env Setup

Install packages under conda env
```bash
conda create -n graphMVP python=3.7
conda activate graphMVP
conda install -y -c rdkit rdkit
conda install -y -c pytorch pytorch=1.7.0
# https://pytorch.org/get-started/previous-versions/
conda install -y numpy networkx scikit-learn

pip install e3fp==1.2.1 msgpack==1.0.0 ipykernel==5.3.0 einops
pip install git+https://github.com/bp-kelley/descriptastorus

export TORCH=1.7.0
export CUDA=cu110  # cu102
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

## GraphMVP
