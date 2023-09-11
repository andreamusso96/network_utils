import networkx as nx
import graph_tool.all as gt
import pandas as pd
import numpy as np


def gt_to_nx(g: gt.Graph) -> nx.Graph:
    assert 'weight' in g.ep, 'Graph does not have edge weights'
    if g.is_directed():
        weighted_adjacency = gt.adjacency(g, weight=g.ep['weight']).toarray().T
        nx_graph = nx.from_numpy_array(weighted_adjacency, create_using=nx.DiGraph)
    else:
        weighted_adjacency = gt.adjacency(g, weight=g.ep['weight']).toarray()
        nx_graph = nx.from_numpy_array(weighted_adjacency).to_undirected()

    _set_node_attributes(g, nx_graph)
    _set_edge_attributed(g, nx_graph)
    _set_graph_attributes(g, nx_graph)
    return nx_graph


def _set_node_attributes(gGt: gt.Graph, gNx: nx.Graph):
    for prop in gGt.vp:
        nx.set_node_attributes(G=gNx, values={int(v): gGt.vp[prop][v] for v in gGt.vertices()}, name=prop)


def _set_edge_attributed(gGt: gt.Graph, gNx: nx.Graph):
    for prop in gGt.ep:
        nx.set_edge_attributes(G=gNx, values={(int(e.source()), int(e.target())): gGt.ep[prop][e] for e in gGt.edges()}, name=prop)


def _set_graph_attributes(gGt: gt.Graph, gNx: nx.Graph):
    for prop in gGt.gp:
        gNx.graph[prop] = gGt.gp[prop]


def nx_to_gt(g: nx.Graph) -> gt.Graph:
    gGt = gt.Graph(directed=False)
    gGt.add_vertex(len(g.nodes))
    gGt.add_edge_list([(int(e[0]), int(e[1])) for e in g.edges])
    _set_vertex_properties(gGt, g)
    _set_edge_properties(gGt, g)
    _set_graph_properties(gGt, g)
    return gGt


def _set_vertex_properties(gGt: gt.Graph, gNx: nx.Graph):
    node_attr = list(set(k for attr_dict in gNx.nodes.data()._nodes.values() for k in attr_dict.keys()))
    df_attributes = pd.DataFrame.from_dict({k: nx.get_node_attributes(gNx, k) for k in node_attr})
    for col in df_attributes.columns:
        gGt.vp[col] = gGt.new_vertex_property(vals=df_attributes[col].values, value_type=_get_prop_type(dtype=df_attributes[col].dtype))


def _set_edge_properties(gGt: gt.Graph, gNx: nx.Graph):
    weights = nx.get_edge_attributes(gNx, 'weight')
    gGt.ep['weight'] = gGt.new_edge_property(vals=list(weights.values()), value_type='float')
    return gGt


def _set_graph_properties(gGt: gt.Graph, gNx: nx.Graph):
    for prop in gNx.graph:
        gGt.gp[prop] = gGt.new_graph_property(val=gNx.graph[prop], value_type='string')


def _get_prop_type(dtype):
    if np.issubdtype(dtype, np.integer):
        return 'int'
    elif np.issubdtype(dtype, float):
        return 'float'
    elif np.issubdtype(dtype, bool):
        return 'bool'
    else:
        return 'string'
