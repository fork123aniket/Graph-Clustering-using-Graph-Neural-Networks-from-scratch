# Spectral Modularity Maximization for Graph Clustering using Graph Neural Networks

This repository implements a graph pooling operator to either coarsen the graph or cluster the similar nodes of the graph together using Spectral Modularity Maximization formulation. This operator is expected to learn the cluster assignment matrix using Graph Neural Networks by the following operations:
```math
\mathbf{S} = \mathrm{softmax}(\mathrm{GCN}(\mathbf{\tilde{A}}, \mathbf{X}))
```
```math
\mathbf{X}^{\prime} = {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
	        \mathbf{X}
```
```math
\mathbf{A}^{\prime} = {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
	        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

```
where $\mathbf{X} \in \mathbb{R}^{N \times F}$ is the node feature matrix of the input graph, $\tilde{A}$ = $\mathbf{D}^{\frac{-1}{2}}\mathbf{AD}^{\frac{-1}{2}}$ is symmetrically normalized adjacency matrix, $\mathbf{A} \in \mathbb{R}^{N \times N}$ is the adjacency matrix of the input graph, $\mathbf{D} = diag(\mathbf{A}1_N)$ is the degree matrix, and $\mathbf{S} \in \mathbb{R}^{BS \times N \times C}$ is the dense learned assignment matrix. The following losses are being used to implement graph clustering and pooling operations:

***Spectral Loss:***
```math
\mathcal{L}_s = - \frac{1}{2m}
	        \cdot{\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{B} \mathbf{S})}

```
where $\mathbf{B}$ is the modularity matrix of the input graph with degree d and m no. of edges, which is defined as:
```math
\mathbf{B} = \mathbf{A} - \frac{dd^{\top}}{2m}
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
	        {\left\|\sum_i\mathbf{S_i}^{\top}\right\|}_F - 1

```
where N is the number of nodes, F is the number of node features, BS is the Batch Size, C is the number of clusters. More in-depth information about this implementation can be found on [PyTorch Geometric Official Website](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.dense.DMoNPooling.html#torch_geometric.nn.dense.DMoNPooling).

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
