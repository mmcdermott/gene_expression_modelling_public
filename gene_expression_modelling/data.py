import numpy as np

from .constants import *

# TODO(mmd): Move some stuff over to pandas constructions
def _level_in(df_idx, conditions={}):
    valid_indices = np.array([True] * len(df_idx))
    for level, allowed_elements in conditions.items():
        if allowed_elements != []: valid_indices &= df_idx.get_level_values(level).isin(allowed_elements)
    return valid_indices

def restrict(
    df,
    perturbagens = [],
    doses        = [],
    durations    = [],
    genes        = [],
    cells        = [],
):
    valid_rows = _level_in(df.index,
        {PERT_NAME_COL: perturbagens, PERT_DOSE_COL: doses, PERT_DRTN_COL: durations, CELL_ID_COL: cells})
    valid_cols = _level_in(df.columns, {GENE_NAME_COL: genes})

    return df.loc[valid_rows, valid_cols]

def _unique_at_level(df_idx, level): return np.unique(df_idx.get_level_values(level))

def unique_perturbagens(df): return _unique_at_level(df.index, PERT_NAME_COL)
def unique_cells(df): return _unique_at_level(df.index, CELL_ID_COL)
def unique_genes(df): return _unique_at_level(df.columns, GENE_NAME_COL)

def unique_doses(df, perturbagens=[]):
    return _unique_at_level(restrict(df, perturbagens=perturbagens).index, PERT_DOSE_COL)
def unique_durations(df, perturbagens=[]):
    return _unique_at_level(restrict(df, perturbagens=perturbagens).index, PERT_DRTN_COL)

def get_num_samples(df): return df.shape[0]
def get_num_genes(df): return df.shape[1]
