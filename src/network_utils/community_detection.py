from typing import Dict, Callable, Union
from enum import Enum

import graph_tool.all as gt
import networkx as nx
import numpy as np
from sklearn.cluster import HDBSCAN, SpectralClustering
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import leidenalg
from scipy import sparse

from .converter import convert_graph, GraphType


class Algorithm(Enum):
    SBM = 'sbm'
    ASSORTATIVE_SBM = 'assortative_sbm'
    NESTED_SBM = 'nested_sbm'
    LOUVAIN = 'louvain'
    LABEL_PROPAGATION = 'label_propagation'
    FLUID = 'fluid'
    HDBSCAN = 'hdbscan'
    GEOGRAPHIC = 'geographic'
    LEIDEN = 'leiden'
    RECURSIVE_SPECTRAL_BISECTION = 'recursive_spectral_bisection'

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
        self.community_id_to_vertex_ids_map = {community: np.array([vertex_id for vertex_id, community_id in self.vertex_id_to_community_id_map.items() if community_id == community]) for community in
                                               self.get_community_ids()}

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

    def get_clustered_adjacency(self, threshold_community_size: int = 10):
        adjacency_matrix = pd.DataFrame(gt.adjacency(self.g, weight=self.g.ep.weight).toarray(), index=np.array(self.g.vp.id.a), columns=np.array(self.g.vp.id.a))
        vertex_ids_communities = np.concatenate(
            [self.get_vertex_ids_community(community) for community in self.get_community_ids() if len(self.get_vertex_ids_community(community)) > threshold_community_size])
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
            cells=dict(values=[
                ['Number of communities', 'Number of communities with one node', 'Number of communities with more than one node', 'Size of largest community', 'Share of largest community',
                 'Size of 10 largest communities'],
                [number_of_communities, number_of_communities_with_one_node, number_of_communities_with_more_than_one_node, size_largest_community, np.round(share_largest_community, decimals=3),
                 size_10_largest_communities]],
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

    def plot_block_matrix(self, lower_level: int = 0, upper_level: int = 5, show: bool = True) -> go.Figure:
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
    elif algorithm == Algorithm.LEIDEN:
        return leiden(g=g, **kwargs)
    elif algorithm == Algorithm.RECURSIVE_SPECTRAL_BISECTION:
        return recursive_spectral_bisection(g=g, **kwargs)
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


def nested_sbm(g: gt.Graph, degree_corrected: bool = True) -> HierarchicalCommunityDetectionResult:
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


def label_propagation(g: gt.Graph) -> CommunityDetectionResult:
    return _apply_networkx_community_detection_algorithm(g=g, algorithm=nx.algorithms.community.asyn_lpa_communities, weight='weight', algorithm_name=Algorithm.LABEL_PROPAGATION.value)


def fluid(g: gt.Graph, n_communities: int) -> CommunityDetectionResult:
    assert g.is_directed() is False, 'Fluid community detection does not support directed graphs'
    return _apply_networkx_community_detection_algorithm(g=g, algorithm=nx.algorithms.community.asyn_fluidc, algorithm_name=Algorithm.FLUID.value, k=n_communities)


def _apply_networkx_community_detection_algorithm(g: gt.Graph, algorithm: Callable, algorithm_name: str, **kwargs) -> CommunityDetectionResult:
    nx_graph = convert_graph(g=g, type_from=GraphType.GT, type_to=GraphType.NX)
    comm = algorithm(nx_graph, **kwargs)
    vertex_ids = nx.get_node_attributes(nx_graph, name='id')
    com_map = {vertex_ids[v]: i for i, c in enumerate(comm) for v in c}
    result = CommunityDetectionResult(g=g, community_map=com_map, algorithm=algorithm_name)
    return result


def hdbscan(g: gt.Graph, min_samples: int = 5, **kwargs) -> CommunityDetectionResult:
    assert g.is_directed() is False, 'HDBSCAN does not support directed graphs'
    weighted_adjacency = gt.adjacency(g, weight=g.ep['weight']).toarray().astype(float)
    inverse_weighted_adjacency = np.divide(1, weighted_adjacency, out=np.inf * np.ones_like(weighted_adjacency), where=weighted_adjacency != 0)
    clusters = HDBSCAN(min_cluster_size=min_samples, min_samples=min_samples, metric="precomputed", n_jobs=-1, **kwargs).fit(inverse_weighted_adjacency)
    com_map = {g.vp.id[v]: clusters.labels_[i] for i, v in enumerate(g.get_vertices())}
    result = CommunityDetectionResult(g=g, community_map=com_map, algorithm=Algorithm.HDBSCAN.value)
    return result


def geographic(g: gt.Graph, lon: float = 109.7) -> CommunityDetectionResult:
    assert 'lon' in g.vp, 'Geographic system detection requires a vertex property "lon", i.e., the longitude of node'
    lon_array = np.array(g.vp['lon'].a)
    communities = np.where(lon_array > -lon, 0, 1)
    com_map = {g.vp.id[v]: communities[i] for i, v in enumerate(g.vertices())}
    result = CommunityDetectionResult(g=g, community_map=com_map, algorithm=Algorithm.GEOGRAPHIC.value)
    return result


def leiden(g: gt.Graph, resolution: float = 1, n_iterations: int = 10) -> CommunityDetectionResult:
    g_ig = convert_graph(g=g, type_from=GraphType.GT, type_to=GraphType.IG)
    partition = leidenalg.find_partition(graph=g_ig, partition_type=leidenalg.RBConfigurationVertexPartition, weights=g_ig.es['weight'], resolution_parameter=resolution, n_iterations=n_iterations)
    com_map = {v['id']: partition.membership[v.index] for v in g_ig.vs()}
    result = CommunityDetectionResult(g=g, community_map=com_map, algorithm=Algorithm.LEIDEN.value)
    return result


def recursive_spectral_bisection(g: gt.Graph, max_value_cut: float, min_size_community: int = 10, discount_factor: float = 1):
    assert g.is_directed() is False, 'Recursive bisection does not support directed graphs'
    nx_graph = convert_graph(g=g, type_from=GraphType.GT, type_to=GraphType.NX)
    community_subgraphs = sum([_recursive_spectral_bisection(g=nx_graph.subgraph(subgraph_vertices), max_value_cut=max_value_cut, min_size_community=min_size_community, discount_factor=discount_factor) for subgraph_vertices in
                               nx.connected_components(nx_graph)], [])
    com_map = {vid: i for i, subgraph in enumerate(community_subgraphs) for vid in nx.get_node_attributes(subgraph, name='id').values()}
    result = CommunityDetectionResult(g=g, community_map=com_map, algorithm=Algorithm.RECURSIVE_SPECTRAL_BISECTION.value)
    return result


def _recursive_spectral_bisection(g: nx.Graph, max_value_cut: float, min_size_community: int, discount_factor: float):
    if g.number_of_nodes() < min_size_community:
        return [g]

    adjacency_matrix = nx.adjacency_matrix(g, weight='weight').todense()
    spectral_clustering = SpectralClustering(n_clusters=2, affinity='precomputed', n_init=100, assign_labels='cluster_qr', n_jobs=-1)
    spectral_clustering.fit(adjacency_matrix)
    labels = spectral_clustering.labels_
    size_0, size_1 = np.sum(labels == 0), np.sum(labels == 1)

    if size_0 == 0 or size_1 == 0:
        return [g]

    nodes = np.array(g.nodes())
    S, T = nodes[labels == 0], nodes[labels == 1]
    cut_value = nx.normalized_cut_size(G=g, S=S, T=T, weight='weight')

    if cut_value < max_value_cut:
        S_graph, T_graph = g.subgraph(S), g.subgraph(T)
        connected_components_S = [g.subgraph(c) for c in nx.connected_components(S_graph)]
        connected_components_T = [g.subgraph(c) for c in nx.connected_components(T_graph)]
        connected_components = connected_components_S + connected_components_T
        return sum([_recursive_spectral_bisection(g=subgraph, max_value_cut=discount_factor * (max_value_cut - cut_value), min_size_community=min_size_community, discount_factor=discount_factor) for subgraph in connected_components], [])
    else:
        return [g]
