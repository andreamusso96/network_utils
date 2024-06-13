import pandas as pd
import numpy as np


def monopartite_projection_rca(edge_list: pd.DataFrame, onto: str, other: str, weight: str) -> pd.DataFrame:
    assert onto in data.columns, f'{onto} not in data columns'
    assert other in data.columns, f'{other} not in data columns'
    assert weight in data.columns, f'{weight} not in data columns'

    m = edge_list.pivot(index=onto, columns=other, values=weight).fillna(0)
    num = m.div(m.sum(axis=1), axis=0)
    den = m.sum(axis=0).div(m.sum().sum())
    rca = num.div(den)
    rca = pd.DataFrame((rca.values > 1).astype(int), columns=rca.columns, index=rca.index)

    num = rca.dot(rca.T)
    den = np.maximum.outer(rca.sum(axis=1).values, rca.sum(axis=1).values)
    adjacency = num.div(den)
    return adjacency