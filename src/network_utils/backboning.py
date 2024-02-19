from typing import Dict, Any, Union
from enum import Enum

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go


class BackBoneMethod(Enum):
    NO = 'no_filter'
    DISPARITY = 'disparity'
    MAXIMUM_SPANNING_TREE = 'maximum_spanning_tree'
    DISPARITY_AND_MAXIMUM_SPANNING_TREE = 'disparity_and_maximum_spanning_tree'


class BackBoneResult:
    def __init__(self, g: nx.Graph, backbone_g: nx.Graph, method: BackBoneMethod, kwargs: Dict[str, Any] = None):
        self.g = g
        self.adjacency_g = pd.DataFrame(nx.adjacency_matrix(g, weight='weight').toarray(), index=g.nodes, columns=g.nodes)
        self.backbone_g = backbone_g
        self.backbone_adjacency_g = pd.DataFrame(nx.adjacency_matrix(backbone_g, weight='weight').toarray(), index=backbone_g.nodes, columns=backbone_g.nodes)
        self.method = method
        self.kwargs = kwargs

    def plot_summary_table(self):
        total_weight_original_graph = np.sum(self.adjacency_g)
        total_weight_backbone_graph = np.sum(self.backbone_adjacency_g)

        total_edges_original_graph = self.g.number_of_edges()
        total_edges_backbone_graph = self.backbone_g.number_of_edges()

        total_node_connected_components_original_graph = self.g.number_of_nodes()
        total_node_connected_components_backbone_graph = np.max([len(a) for a in nx.connected_components(self.backbone_g)])

        average_degree_original_graph = total_edges_original_graph / total_node_connected_components_original_graph
        average_degree_backbone_graph = total_edges_backbone_graph / total_node_connected_components_backbone_graph

        table = go.Figure(data=[go.Table(
            header=dict(values=['', 'Original', 'Backbone', 'Ratio'],
                        line_color='darkslategray',
                        fill_color='lightskyblue',
                        align='left'),
            cells=dict(values=[['Total weight', 'Total edges', 'Nodes in LCC', 'Average degree'],
                               [total_weight_original_graph, total_edges_original_graph, total_node_connected_components_original_graph, np.round(average_degree_original_graph, decimals=3)],
                               [total_weight_backbone_graph, total_edges_backbone_graph, total_node_connected_components_backbone_graph, np.round(average_degree_backbone_graph, decimals=3)],
                               [np.round(total_weight_backbone_graph / total_weight_original_graph, decimals=3),
                                np.round(total_edges_backbone_graph / total_edges_original_graph, decimals=3),
                                np.round(total_node_connected_components_backbone_graph / total_node_connected_components_original_graph, decimals=3),
                                np.round(average_degree_backbone_graph / average_degree_original_graph, decimals=3)]],
                       line_color='darkslategray',
                       fill_color='lightcyan',
                       align='left'))
        ])
        table.show(renderer='browser')


def get_network_backbone(adj: pd.DataFrame = None, g: nx.Graph = None, method: BackBoneMethod = BackBoneMethod, directed: bool = None, **kwargs) -> BackBoneResult:
    assert adj is not None or g is not None, 'Either adjacency or graph must be provided'
    assert adj is None or g is None, 'Either adjacency or graph must be provided, but not both'

    if adj is not None:
        assert directed is not None, 'Directed must be provided when adjacency is provided'
        assert np.max(np.diag(adj.values)) == 0, 'Adjacency matrix has self-loops, no self-loops are allowed'
        return _get_network_backbone_from_adjacency(adj=adj, method=method, directed=directed, **kwargs)
    else:
        return _get_network_backbone_from_graph(g=g, method=method, **kwargs)


def _get_network_backbone_from_adjacency(adj: pd.DataFrame, method: BackBoneMethod, directed: bool, **kwargs) -> BackBoneResult:
    if directed:
        g = nx.from_pandas_adjacency(adj, create_using=nx.DiGraph)
    else:
        g = nx.from_pandas_adjacency(adj, create_using=nx.Graph)

    return _get_network_backbone_from_graph(g=g, method=method, **kwargs)


def _get_network_backbone_from_graph(g: nx.Graph, method: BackBoneMethod, **kwargs) -> BackBoneResult:
    if method == BackBoneMethod.NO:
        backbone_g = g
    elif method == BackBoneMethod.DISPARITY:
        backbone_g = disparity_edge_filter(g=g, **kwargs)
    else:
        raise ValueError(f'Unknown backbone method {method}')

    return BackBoneResult(g=g, backbone_g=backbone_g, method=method, kwargs=kwargs)


def disparity_edge_filter(g: nx.Graph, alpha: float = 0.05) -> nx.Graph:
    weighted_adjacency_graph = nx.adjacency_matrix(g, weight='weight').toarray()
    p_values_in_edges = _compute_pvalues_edges(weighted_adjacency_graph=weighted_adjacency_graph.T)
    p_values_out_edges = _compute_pvalues_edges(weighted_adjacency_graph=weighted_adjacency_graph)
    p_values_edges = np.minimum(p_values_in_edges, p_values_out_edges)
    adjacency_filtered_graph = p_values_edges < alpha
    weighted_adjacency_filtered_graph = np.multiply(weighted_adjacency_graph, adjacency_filtered_graph)
    filtered_graph = nx.from_pandas_adjacency(pd.DataFrame(weighted_adjacency_filtered_graph, index=g.nodes, columns=g.nodes), create_using=type(g))
    return filtered_graph

def _compute_pvalues_edges(weighted_adjacency_graph: np.ndarray) -> np.ndarray:
    node_strengths = weighted_adjacency_graph.sum(axis=1).reshape(-1, 1)
    normalized_adjacency_graph = np.divide(weighted_adjacency_graph, node_strengths, out=np.zeros_like(weighted_adjacency_graph), where=node_strengths != 0)
    degrees = (weighted_adjacency_graph > 0).sum(axis=1)
    degrees_min_one = degrees - 1
    p_values_edges = np.power(1 - normalized_adjacency_graph, degrees_min_one, out=np.ones_like(weighted_adjacency_graph), where=degrees_min_one > 0)
    return p_values_edges