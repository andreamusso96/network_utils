from typing import Dict, Any
from enum import Enum

import numpy as np
import pandas as pd
import graph_tool.all as gt
import plotly.graph_objects as go

from . import adjacency


class BackBoneMethod(Enum):
    NO = 'no_filter'
    DISPARITY = 'disparity'
    MAXIMUM_SPANNING_TREE = 'maximum_spanning_tree'


class BackBoneResult:
    def __init__(self, g: gt.Graph, backbone_g: gt.Graph, method: BackBoneMethod, kwargs: Dict[str, Any] = None):
        self.g = g
        self.backbone_g = backbone_g
        self.method = method
        self.kwargs = kwargs

    def plot_heterogeneity_of_edge_weights(self):
        weighted_adjacency_graph = gt.adjacency(self.g, weight=self.g.ep['weight']).toarray().astype(float)
        node_strengths = weighted_adjacency_graph.sum(axis=1)
        normalized_adjacency_graph = np.divide(weighted_adjacency_graph, node_strengths, out=np.zeros_like(weighted_adjacency_graph), where=node_strengths != 0)
        degrees = (weighted_adjacency_graph > 0).sum(axis=1)
        heterogeneity = degrees * np.sum(np.square(normalized_adjacency_graph), axis=1)
        fig = go.Figure(data=go.Scatter(x=degrees, y=heterogeneity, mode='markers'))
        fig.update_layout(title=f'Heterogeneity of edge weights for {self.method.value} backbone')
        fig.update_xaxes(title='Degree', type='log')
        fig.update_yaxes(title='Heterogeneity', type='log')
        fig.show(renderer='browser')

    def plot_summary_table(self):
        total_weight_original_graph = np.sum(np.array(self.g.ep['weight'].a))
        total_weight_backbone_graph = np.sum(np.array(self.backbone_g.ep['weight'].a))

        total_edges_original_graph = self.g.num_edges()
        total_edges_backbone_graph = self.backbone_g.num_edges()

        total_node_connected_components_original_graph = self.g.num_vertices()
        total_node_connected_components_backbone_graph = gt.extract_largest_component(self.backbone_g, directed=False).num_vertices()

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


def get_network_backbone(g: gt.Graph, method: BackBoneMethod, **kwargs) -> BackBoneResult:
    assert 'id' in g.vp, 'Graph does not have node ids'
    assert 'weight' in g.ep, 'Graph does not have edge weights'

    if method == BackBoneMethod.NO:
        backbone_g = g
    elif method == BackBoneMethod.DISPARITY:
        backbone_g = disparity_edge_filter(g=g, **kwargs)
    elif method == BackBoneMethod.MAXIMUM_SPANNING_TREE:
        backbone_g = maximum_spanning_tree_edge_filter(g=g)
    else:
        raise ValueError(f'Unknown backbone method {method}')

    return BackBoneResult(g=g, backbone_g=backbone_g, method=method, kwargs=kwargs)


def disparity_edge_filter(g: gt.Graph, alpha: float = 0.05) -> gt.Graph:
    weighted_adjacency_graph = gt.adjacency(g, weight=g.ep['weight']).toarray().astype(float)
    p_values_in_edges = _compute_pvalues_edges(weighted_adjacency_graph=weighted_adjacency_graph)
    p_values_out_edges = _compute_pvalues_edges(weighted_adjacency_graph=weighted_adjacency_graph.T)
    p_values_edges = np.minimum(p_values_in_edges, p_values_out_edges)
    adjacency_filtered_graph = p_values_edges < alpha
    weighted_adjacency_filtered_graph = np.multiply(weighted_adjacency_graph, adjacency_filtered_graph)
    filtered_graph = _get_filtered_graph_from_weighted_adjacency_filtered_graph(weighted_adjacency_filtered_graph=weighted_adjacency_filtered_graph, vertex_ids=np.array(g.vp['id'].a),
                                                                                filter_name='disparity', directed=g.is_directed())
    return filtered_graph


def maximum_spanning_tree_edge_filter(g: gt.Graph) -> gt.Graph:
    maximum_spanning_tree = gt.min_spanning_tree(g, weights=g.new_edge_property('double', vals=-1*g.ep['weight'].a))
    filtered_graph = gt.Graph(gt.GraphView(g, efilt=maximum_spanning_tree), prune=True)
    filtered_graph.gp['filter'] = filtered_graph.new_graph_property(value_type="string", val='maximum_spanning_tree')
    return filtered_graph

def _compute_pvalues_edges(weighted_adjacency_graph: np.ndarray):
    node_strengths = weighted_adjacency_graph.sum(axis=1).reshape(-1, 1)
    normalized_adjacency_graph = np.divide(weighted_adjacency_graph, node_strengths, out=np.zeros_like(weighted_adjacency_graph), where=node_strengths != 0)
    degrees = (weighted_adjacency_graph > 0).sum(axis=1)
    degrees_min_one = degrees - 1
    p_values_edges = np.power(1 - normalized_adjacency_graph, degrees_min_one, out=np.ones_like(weighted_adjacency_graph), where=degrees_min_one > 0)
    return p_values_edges


def _get_filtered_graph_from_weighted_adjacency_filtered_graph(weighted_adjacency_filtered_graph: np.ndarray, vertex_ids: np.ndarray, filter_name: str, directed: bool) -> gt.Graph:
    weighted_adjacency_filtered_graph = pd.DataFrame(weighted_adjacency_filtered_graph, index=vertex_ids, columns=vertex_ids)
    if directed:
        filtered_graph = adjacency.directed_weighted_graph_from_weight_matrix(weight_matrix=weighted_adjacency_filtered_graph)
    else:
        filtered_graph = adjacency.undirected_weighted_graph_from_weight_matrix(weight_matrix=weighted_adjacency_filtered_graph)

    filtered_graph.gp['filter'] = filtered_graph.new_graph_property(value_type="string", val=filter_name)
    return filtered_graph
