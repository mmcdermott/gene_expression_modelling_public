# Adapted from https://github.com/cmap/cmapPy/blob/master/cmapPy/pandasGEXpress/parse_gctx.py
# to work with python 3.6, and to add some additional features.

import os, h5py, numpy as np, pandas as pd
from .constants import *
from .dose import Dose

DATA_NODE           = '/0/DATA/0/matrix'
ROW_META_GROUP_NODE = '/0/META/ROW/id'
COL_META_GROUP_NODE = '/0/META/COL/id'

def _is_gctx_file(f): return f.endswith('.gctx')
def _is_inst_info(f): return 'inst_info' in f
def _is_pert_info(f): return 'pert_info' in f
def _is_cell_info(f): return 'cell_info' in f
def _is_gene_info(p): return lambda f: p in f # TODO(mmd): don't be stupid

# TODO(mmd): Rethink doses.
# TODO(mmd): Rethink skip_first_col nonsense. Work with names.

def parse(
    directory, file_specifier=None,
    gene_info_index_col = 'gene_id', # TODO(mmd): Does this default make sense?
    gene_info_file_part = 'gene_info',
):
    """
    Parses a .gctx file into a multi-indexed pandas DataFrame.
    Input: gctx_path (str): full path to gctx file you want to parse.
    Output: Pandas dataframe containing the data, with multi-indices on the rows and columns as appropriate.
    """
    assert os.path.isdir(directory)

    files = os.listdir(directory)
    prepend = lambda fs: list(map(lambda f: os.path.join(directory, f), fs))
    if file_specifier is None:
        gctx_files = prepend(filter(_is_gctx_file, files))
    else:
        gctx_files = prepend([file_specifier])
    inst_info_files = prepend(filter(_is_inst_info, files))
    pert_info_files = prepend(filter(_is_pert_info, files))
    cell_info_files = prepend(filter(_is_cell_info, files))
    gene_info_files = prepend(filter(_is_gene_info(gene_info_file_part), files))

    assert len(gctx_files) == 1,      'There must be exactly 1 *.gctx file in this directory'
    assert len(inst_info_files) == 1, 'There must be exactly 1 *inst_info* file in this directory'
    assert len(pert_info_files) == 1, 'There must be exactly 1 *pert_info* file in this directory'
    assert len(cell_info_files) == 1, 'There must be exactly 1 *cell_info* file in this directory'
    assert len(gene_info_files) == 1, 'There must be exactly 1 *gene_info* file in this directory'

    gctx_path = gctx_files[0]
    inst_info = _parse_tsv(inst_info_files[0], index_col='inst_id')
    pert_info = _parse_tsv(pert_info_files[0], index_col='pert_id')
    cell_info = _parse_tsv(cell_info_files[0], index_col='cell_id')
    gene_info = _parse_tsv(gene_info_files[0], index_col=gene_info_index_col)
    inst_cols, pert_cols, cell_cols = (set(df.columns) for df in (inst_info, pert_info, cell_info))

    samples_info = inst_info.join(
        pert_info.filter(items = pert_cols-inst_cols), on='pert_id'
    ).join(
        cell_info.filter(items = cell_cols-inst_cols), on='cell_id'
    )

    with h5py.File(gctx_path, "r") as gctx_file:
        row_ids = _read_h5_group(gctx_file[ROW_META_GROUP_NODE]).astype(int)
        col_ids = _read_h5_group(gctx_file[COL_META_GROUP_NODE]).astype(str)
        data_array = _read_h5_group(gctx_file[DATA_NODE]).T

        data_df_raw = pd.DataFrame(data_array, index=row_ids, columns=col_ids)
        data_df_raw.index.set_names('pr_gene_id', inplace=True)
        data_df_raw.columns.set_names('inst_id', inplace=True)

        data_df = data_df_raw.join(gene_info).set_index(list(gene_info.columns), append=True).T
        data_df.index = pd.DataFrame(
            index=data_df.index
        ).join(
            samples_info
        ).set_index(
            list(samples_info.columns), append=True
        ).index

        data_df = _normalize_index(_fix(data_df), GSE_COLUMN_MAPPINGS)

    return data_df

def _read_h5_group(h):
    arr = np.empty(h.shape, dtype=h.dtype)
    h.read_direct(arr)
    return arr

def _normalize_index(df, idx_mapping):
    for new, old in idx_mapping.items(): df.index.rename([new], [old], inplace=True)

    doses = list(df.index.get_level_values(PERT_DOSE_COL))
    units = list(df.index.get_level_values(PERT_DOSE_UNIT_COL))
    df.index = df.index.droplevel(PERT_DOSE_COL).droplevel(PERT_DOSE_UNIT_COL)

    #df[PERT_DOSE_COL] = pd.Series([Dose(dose, unit) for dose, unit in zip(doses, units)], index=df.index)
    df[PERT_DOSE_COL] = pd.Series(['{u}@{d}'.format(d=d, u=u) for d, u in zip(doses, units)], index=df.index)
    df.set_index(PERT_DOSE_COL, append=True, inplace=True)

    return df

def _parse_tsv(path, index_col=None):
    df = pd.read_csv(path, header=0, sep='\t')
    if index_col is not None: df.set_index(index_col, inplace=True)
    return df

def _fix(df):
    return df.apply(
        lambda x: pd.to_numeric(x, errors="ignore")
    ).applymap(
        lambda x: np.NaN if x in NAN_VALUES else x
    )
