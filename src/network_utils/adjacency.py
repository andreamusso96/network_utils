import numpy as np
import pandas as pd
import graph_tool.all as gt


def weighted_graph_from_weight_matrix(weight_matrix: pd.DataFrame) -> gt.Graph:
    symmetric_weight_matrix = (weight_matrix + weight_matrix.T) / 2
    mask = symmetric_weight_matrix.values.nonzero()
    weights = symmetric_weight_matrix.values[mask]
    g = gt.Graph(directed=False)
    g.add_edge_list(np.transpose(mask))
    g.ep['weight'] = g.new_edge_property("double", vals=weights)
    g.vp['id'] = g.new_vertex_property("int", vals=weight_matrix.index)
    gt.remove_parallel_edges(g=g)
    return g
