from typing import Dict, Callable, Union
from enum import Enum

import graph_tool.all as gt
import networkx as nx
import numpy as np
from sklearn.cluster import HDBSCAN
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . gt_to_nx import gt_to_nx


class Algorithm(Enum):
    SBM = 'sbm'
    ASSORTATIVE_SBM = 'assortative_sbm'
    NESTED_SBM = 'nested_sbm'
    LOUVAIN = 'louvain'
    LABEL_PROPAGATION = 'label_propagation'
    FLUID = 'fluid'
    HDBSCAN = 'hdbscan'
    GEOGRAPHIC = 'geographic'

    @staticmethod
    def from_string(algorithm_str: str):
        for algorithm in Algorithm:
            if algorithm.value == algorithm_str:
                return algorithm
        raise ValueError(f"Unknown algorithm {algorithm_str}")


class CommunityDetectionResult:
    def __init__(self, g: gt.Graph, community_map: Dict[int, int], algorithm: str):
        self.g = g
        self.algorithm = algorithm
        self.vertex_id_to_community_id_map = community_map
        self.community_id_to_vertex_ids_map = {community: np.array([vertex_id for vertex_id, community_id in self.vertex_id_to_community_id_map.items() if community_id == community]) for community in self.get_community_ids()}

    def get_vertex_ids_community(self, community: int) -> np.ndarray:
        return self.community_id_to_vertex_ids_map[community]

    def get_community_ids(self) -> np.ndarray:
        return np.unique(sorted(list(self.vertex_id_to_community_id_map.values())))

    def get_community_id_vertex(self, vertex_id: int) -> int:
        return self.vertex_id_to_community_id_map[vertex_id]

    def plot_block_matrix(self, show: bool = True, threshold_community_size: int = 5) -> go.Figure:
        block_matrix = self.get_block_matrix()
        communities_with_size_above_threshold = [community for community in self.get_community_ids() if len(self.get_vertex_ids_community(community)) > threshold_community_size]
        block_matrix = block_matrix.loc[communities_with_size_above_threshold, communities_with_size_above_threshold]
        fig = go.Figure(data=go.Heatmap(z=block_matrix.values, x=[str(a) for a in block_matrix.columns], y=[str(a) for a in block_matrix.index]))
        fig.update_layout(template='plotly_white', title_text="Block matrices")
        if show:
            fig.show(renderer="browser")
            return fig
        else:
            return fig

    def get_block_matrix(self) -> pd.DataFrame:
        edge_list_with_weights = [(self.g.vp.id[e.source()], self.g.vp.id[e.target()], self.g.ep.weight[e]) for e in self.g.edges()]
        edge_list_with_weights = pd.DataFrame(edge_list_with_weights, columns=['source', 'target', 'weight'])
        edge_list_with_weights['source_community'] = edge_list_with_weights['source'].apply(lambda x: self.vertex_id_to_community_id_map[x])
        edge_list_with_weights['target_community'] = edge_list_with_weights['target'].apply(lambda x: self.vertex_id_to_community_id_map[x])
        block_matrix = edge_list_with_weights.groupby(['source_community', 'target_community'])['weight'].sum().unstack().fillna(0)
        return block_matrix

    def get_clustered_adjacency(self, threshold_community_size: int = 130):
        adjacency_matrix = pd.DataFrame(gt.adjacency(self.g, weight=self.g.ep.weight).toarray(), index=np.array(self.g.vp.id.a), columns=np.array(self.g.vp.id.a))
        vertex_ids_communities = np.concatenate([self.get_vertex_ids_community(community) for community in self.get_community_ids() if len(self.get_vertex_ids_community(community)) > threshold_community_size])
        adjacency_matrix = adjacency_matrix.loc[vertex_ids_communities, vertex_ids_communities]
        return adjacency_matrix

    def plot_summary_table(self):
        number_of_communities = len(self.get_community_ids())
        community_sizes = [len(self.get_vertex_ids_community(community)) for community in self.get_community_ids()]
        size_largest_community = np.max(community_sizes)
        share_largest_community = size_largest_community / self.g.num_vertices()
        size_10_largest_communities = ", ".join([str(a) for a in np.sort(community_sizes)[::-1][:10]])
        number_of_communities_with_one_node = np.where(np.array(community_sizes) == 1, 1, 0).sum()
        number_of_communities_with_more_than_one_node = number_of_communities - number_of_communities_with_one_node
        table = go.Figure(data=[go.Table(
            header=dict(values=['', self.algorithm],
                        line_color='darkslategray',
                        fill_color='lightskyblue',
                        align='left'),
            cells=dict(values=[['Number of communities', 'Number of communities with one node', 'Number of communities with more than one node', 'Size of largest community', 'Share of largest community', 'Size of 10 largest communities'],
                               [number_of_communities, number_of_communities_with_one_node, number_of_communities_with_more_than_one_node, size_largest_community, np.round(share_largest_community, decimals=3), size_10_largest_communities]],
                       line_color='darkslategray',
                       fill_color='lightcyan',
                       align='left'))
        ])
        table.show(renderer='browser')


class HierarchicalCommunityDetectionResult:
    def __init__(self, g: gt.Graph, hierarchical_community_map: Dict[int, CommunityDetectionResult], algorithm: str):
        self.g = g
        self.algorithm = algorithm
        self.community_map = hierarchical_community_map
        self.levels = list(self.community_map.keys())

    def get_level(self, level: int):
        return self.community_map[level]

    def plot_block_matrix(self, lower_level: int = 0, upper_level: int = 5,  show: bool = True) -> go.Figure:
        block_matrices = {level: self.get_level(level=level).get_block_matrix() for level in self.levels if lower_level <= level <= upper_level}
        fig = make_subplots(rows=1, cols=len(block_matrices), subplot_titles=[f"Level {level}" for level in block_matrices])
        traces = [go.Heatmap(z=block_matrices[level].values, x=block_matrices[level].index, y=block_matrices[level].index, showscale=False) for level in block_matrices]
        for i, trace in enumerate(traces):
            fig.add_trace(trace, row=1, col=i + 1)

        fig.update_layout(template='plotly_white', title_text="Block matrices")
        if show:
            fig.show(renderer="browser")
            return fig
        else:
            return fig


def run_community_detection(g: gt.Graph, algorithm: Algorithm, **kwargs) -> Union[CommunityDetectionResult, HierarchicalCommunityDetectionResult]:
    if algorithm == Algorithm.SBM:
        return sbm(g=g, **kwargs)
    elif algorithm == Algorithm.ASSORTATIVE_SBM:
        return assortative_sbm(g=g)
    elif algorithm == Algorithm.NESTED_SBM:
        return nested_sbm(g=g, **kwargs)
    elif algorithm == Algorithm.LOUVAIN:
        return louvain(g=g)
    elif algorithm == Algorithm.LABEL_PROPAGATION:
        return label_propagation(g=g)
    elif algorithm == Algorithm.FLUID:
        return fluid(g=g, **kwargs)
    elif algorithm == Algorithm.HDBSCAN:
        return hdbscan(g=g, **kwargs)
    elif algorithm == Algorithm.GEOGRAPHIC:
        return geographic(g=g, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm {algorithm}")


def sbm(g: gt.Graph, degree_corrected: bool = True) -> CommunityDetectionResult:
    scale_exponential_distribution = np.array(g.ep['weight'].a).mean()
    state_args = dict(recs=[g.ep['weight']], rec_types=["real-exponential"],
                      rec_params=[{"alpha": 1, "beta": scale_exponential_distribution}], deg_corr=degree_corrected)
    state = gt.minimize_blockmodel_dl(g=g, state=gt.BlockState, state_args=state_args)
    _refine_state(state=state, niter=100)
    com_map = _extract_community_map_from_block_property_map(g=g, block_prop_map=state.get_blocks())
    result = CommunityDetectionResult(g=g, community_map=com_map, algorithm=Algorithm.SBM.value)
    return result


def assortative_sbm(g: gt.Graph) -> CommunityDetectionResult:
    state = gt.minimize_blockmodel_dl(g=g, state=gt.PPBlockState)
    _refine_state(state=state, niter=100)
    com_map = _extract_community_map_from_block_property_map(g=g, block_prop_map=state.get_blocks())
    result = CommunityDetectionResult(g=g, community_map=com_map, algorithm=Algorithm.ASSORTATIVE_SBM.value)
    return result


def nested_sbm(g: gt.Graph, degree_corrected: bool = True):
    base_state_args = dict(state_args=dict(recs=[g.ep['weight']], rec_types=["real-exponential"], deg_corr=degree_corrected))
    state = gt.minimize_nested_blockmodel_dl(g=g, state=gt.NestedBlockState, state_args=base_state_args)
    _refine_state(state=state, niter=100)
    n_levels = len(state.get_levels())
    hcom_map = {}
    label_level, n_communities_previous_level = 0, np.inf
    for level in range(n_levels):
        map_level = _extract_community_map_from_block_property_map(g=g, block_prop_map=state.project_level(level).get_blocks())
        n_communities_level = len(np.unique(list(map_level.values())))
        if 1 < n_communities_level < n_communities_previous_level:
            com_map_level = _extract_community_map_from_block_property_map(g=g, block_prop_map=state.project_level(level).get_blocks())
            hcom_map[label_level] = CommunityDetectionResult(g=g, community_map=com_map_level, algorithm=Algorithm.NESTED_SBM.value)
            n_communities_previous_level = n_communities_level
            label_level += 1

    result = HierarchicalCommunityDetectionResult(g=g, hierarchical_community_map=hcom_map, algorithm=Algorithm.NESTED_SBM.value)
    return result


def _extract_community_map_from_block_property_map(g, block_prop_map):
    block_prop_map = gt.contiguous_map(block_prop_map)
    com_map = {g.vp.id[v]: block_prop_map[v] for v in g.vertices()}
    return com_map


def _refine_state(state, niter: int = 100):
    for i in range(niter):
        state.multiflip_mcmc_sweep(niter=niter, beta=np.inf)


def louvain(g: gt.Graph) -> CommunityDetectionResult:
    return _apply_networkx_community_detection_algorithm(g=g, algorithm=nx.algorithms.community.louvain_communities, weight='weight', algorithm_name=Algorithm.LOUVAIN.value)


def label_propagation(g: gt.Graph):
    return _apply_networkx_community_detection_algorithm(g=g, algorithm=nx.algorithms.community.asyn_lpa_communities, weight='weight', algorithm_name=Algorithm.LABEL_PROPAGATION.value)


def fluid(g: gt.Graph, n_communities: int):
    return _apply_networkx_community_detection_algorithm(g=g, algorithm=nx.algorithms.community.asyn_fluidc, algorithm_name=Algorithm.FLUID.value, k=n_communities)


def _apply_networkx_community_detection_algorithm(g: gt.Graph, algorithm: Callable, algorithm_name: str, **kwargs):
    nx_graph = gt_to_nx(g=g)
    comm = algorithm(nx_graph, **kwargs)
    com_map = {nx.get_node_attributes(nx_graph, name='id')[v]: i for i, c in enumerate(comm) for v in c}
    result = CommunityDetectionResult(g=g, community_map=com_map, algorithm=algorithm_name)
    return result


def hdbscan(g: gt.Graph, min_samples: int = 5, **kwargs):
    weighted_adjacency = gt.adjacency(g, weight=g.ep['weight']).toarray().astype(float)
    inverse_weighted_adjacency = np.divide(1, weighted_adjacency, out=np.inf * np.ones_like(weighted_adjacency), where=weighted_adjacency != 0)
    clusters = HDBSCAN(min_cluster_size=min_samples, min_samples=min_samples, metric="precomputed", n_jobs=-1, **kwargs).fit(inverse_weighted_adjacency)
    com_map = {g.vp.id[v]: clusters.labels_[i] for i, v in enumerate(g.get_vertices())}
    result = CommunityDetectionResult(g=g, community_map=com_map, algorithm=Algorithm.HDBSCAN.value)
    return result


def geographic(g: gt.Graph, lon: float = 109.7):
    assert 'lon' in g.vp, 'Geographic system detection requires a vertex property "lon", i.e., the longitude of node'
    lon_array = np.array(g.vp['lon'].a)
    communities = np.where(lon_array > -lon, 0, 1)
    com_map = {g.vp.id[v]: communities[i] for i, v in enumerate(g.vertices())}
    result = CommunityDetectionResult(g=g, community_map=com_map, algorithm=Algorithm.GEOGRAPHIC.value)
    return result
