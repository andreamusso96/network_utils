import numpy as np
import pandas as pd
import graph_tool.all as gt


def undirected_weighted_graph_from_weight_matrix(weight_matrix: pd.DataFrame) -> gt.Graph:
    symmetric_weight_matrix = (weight_matrix + weight_matrix.T) / 2
    g = _build_graph(weight_matrix=symmetric_weight_matrix, vertex_ids=weight_matrix.index, directed=False)
    return g


def directed_weighted_graph_from_weight_matrix(weight_matrix: pd.DataFrame) -> gt.Graph:
    g = _build_graph(weight_matrix=weight_matrix, vertex_ids=weight_matrix.index, directed=True)
    return g


def _build_graph(weight_matrix: pd.DataFrame, vertex_ids: np.ndarray, directed: bool) -> gt.Graph:
    mask = weight_matrix.values.nonzero()
    weights = weight_matrix.values[mask]
    g = gt.Graph(directed=directed)
    g.add_vertex(n=len(vertex_ids))
    g.add_edge_list(np.transpose(mask))
    g.ep['weight'] = g.new_edge_property("double", vals=weights)
    g.vp['id'] = g.new_vertex_property("int", vals=vertex_ids)
    gt.remove_parallel_edges(g=g)
    return g
