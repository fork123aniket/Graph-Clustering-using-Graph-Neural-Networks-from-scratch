# Spectral Modularity Maximization for Graph Clustering using Graph Neural Networks

This repository implements a graph pooling operator to either coarsen the graph or cluster the similar nodes of the graph together using Spectral Modularity Maximization formulation. This operator is expected to learn the cluster assignment matrix using Graph Neural Networks by the following operations:
```math
\mathbf{X}^{\prime} = {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
	        \mathbf{X}
```
```math
\mathbf{A}^{\prime} = {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
	        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

```
where $\mathbf{S} \in \mathbb{R}^{B \times N \times C}$ is the dense learned assignment matrix. The following losses are being used to implement graph clustering and pooling operations:

***Spectral Loss:***
```math
\mathcal{L}_s = - \frac{1}{2m}
	        \cdot{\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{B} \mathbf{S})}

```
***Orthogonal Loss:***
```math
\mathcal{L}_o = {\left\| \frac{\mathbf{S}^{\top} \mathbf{S}}
	        {{\|\mathbf{S}^{\top} \mathbf{S}\|}_F} -\frac{\mathbf{I}_C}{\sqrt{C}}
	        \right\|}_F

```
***Cluster Loss:***
```math
\mathcal{L}_c = \frac{\sqrt{C}}{n}
	        {\left\|\sum_i\mathbf{C_i}^{\top}\right\|}_F - 1

```
More in-depth information about this implementation can be found on [PyTorch Geometric Official Website](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.dense.DMoNPooling.html#torch_geometric.nn.dense.DMoNPooling).

## Requirements
-	`math`
-	`PyTorch`
-	`PyTorch Geometric`

## Usage

### Data
The `PROTEINS` dataset contains `1113` homogeneous attributed graphs with average no. of nodes (`~39.1`), edges (`~145.6`), features (`3`), and classes (`2`) per graph.
### Training and Testing
-	The layer implementation can be found inside `dmon_pool.py`.
-	To train and test this layer implementation on graphs, run `example.py`, and it prints `train`, `validation`, and `test` accuracies alongside their corresponding losses after every epoch.

### Note
-	Though the example given performs only the graph pooling operation, it is possible to use the GNN layer only to compute the clusters, each having similar nodes, by just calculating the `cluster loss` and turning off the other two losses.
-	Since the implementation is meant to either cluster the nodes or coarsen the graph, the implementation is only to perform `graph-level` tasks such as ***graph classification***.
