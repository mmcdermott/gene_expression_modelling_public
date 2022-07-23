# Adapted from https://github.com/cmap/cmapPy/blob/master/cmapPy/pandasGEXpress/parse_gctx.py
# to work with python 3.6, and to add some additional features.

import os, h5py, numpy as np, pandas as pd
from .constants import *
from .dose import Dose

DATA_NODE           = '/0/DATA/0/matrix'
ROW_META_GROUP_NODE = '/0/META/ROW'
COL_META_GROUP_NODE = '/0/META/COL'

def parse(path, metadata_filenames=(None, None)):
    assert path.endswith('.gctx') or path.endswith('.gct')
    return _parse_gctx(path, metadata_filenames) if path.endswith('.gctx') else _parse_gct(path)

def _normalize_index(df, idx_mapping):
    if len(df.index.names) == 1: return _normalize_uniindex(df, idx_mapping)
    else: return _normalize_multiindex(df, idx_mapping)

def _normalize_uniindex(df, idx_mapping):
        name = df.index.names[0]
        if name in idx_mapping: df.index.rename(idx_mapping[name], inplace=True)

        return df

def _normalize_multiindex(df, idx_mapping):
    for new, old in idx_mapping.items():
        if type(old) == list:
            for o in old:
                try:
                    df.index.rename([new], [o], inplace=True)
                    break
                except KeyError: pass
        else: 
            df.index.rename([new], level=[old], inplace=True)

    if PERT_DOSE_COL in df.index:
        doses = list(df.index.get_level_values(PERT_DOSE_COL))
        units = list(df.index.get_level_values(PERT_DOSE_UNIT_COL))
        df.index = df.index.droplevel(PERT_DOSE_COL).droplevel(PERT_DOSE_UNIT_COL)

        df[PERT_DOSE_COL] = pd.Series([Dose(dose, unit) for dose, unit in zip(doses, units)], index=df.index)
        df.set_index(PERT_DOSE_COL, append=True, inplace=True)

    return df

def _parse_gctx(gctx_path, metadata_filenames=(None, None)):
    """
    Parses a .gctx file into a multi-indexed pandas DataFrame.
    Input: gctx_path (str): full path to gctx file you want to parse.
    Output: Pandas dataframe containing the data, with multi-indices on the rows and columns as appropriate.
    """
    # Open file.
    with h5py.File(gctx_path, "r") as gctx_file:
        row_meta_filename, col_meta_filename = metadata_filenames
        if row_meta_filename is None:
            print('WOOO')
            row_meta = _parse_metadata_df(gctx_file[ROW_META_GROUP_NODE])
            print(row_meta.head())
        else: row_meta = pd.read_csv(row_meta_filename, sep='\t', index_col=0)

        if col_meta_filename is None: col_meta = _parse_metadata_df(gctx_file[COL_META_GROUP_NODE])
        else: col_meta = pd.read_csv(col_meta_filename, sep='\t', index_col=0)

        return _parse_data_df(gctx_file[DATA_NODE], row_meta, col_meta)

def _parse_metadata_df(meta_group):
    """
    Reads in all metadata from .gctx file to a pandas DataFrame.
    Input: meta_group (HDF5 group): Group from which to read metadata values
    Output: meta_df (pandas DataFrame): data frame corresponding to metadata fields of dimension specified.
    """
    # read values from hdf5 & make a DataFrame
    header_values = {}
    for k in meta_group.keys():
        curr_dset = meta_group[k]
        temp_array = np.empty(curr_dset.shape, dtype = curr_dset.dtype)
        curr_dset.read_direct(temp_array)
        header_values[str(k)] = temp_array
    # need to temporarily make dtype of all values str so that to_numeric
    # works consistently with gct vs gctx parser.
    meta_df = pd.DataFrame.from_dict(header_values).astype(str)
    meta_df = meta_df.applymap(lambda x: x[2:-1] if x.startswith("b'") and x.endswith("'") else x)
    return meta_df.apply(lambda x: pd.to_numeric(x, errors="ignore")).replace(NAN_VALUES, np.nan)

def _parse_data_df(data_dset, row_meta, col_meta):
    """
    Parses in data_df from hdf5
    Input:
        data_dset (h5py dset): HDF5 dataset from which to read data_df
        row_meta (pandas DataFrame): the parsed in row metadata
        col_meta (pandas DataFrame): the parsed in col metadata
    Output: pandas DataFrame: data frame corresponding to the extracted data.
    """
    data_array = np.empty(data_dset.shape, dtype=DATA_TYPE)
    data_dset.read_direct(data_array)
    to_idx = lambda X: X.set_index(list(X.columns)).index
    df = pd.DataFrame(data_array.transpose(), index=to_idx(row_meta), columns=to_idx(col_meta))
    return _normalize_index(df.transpose(), CMAP_GCTX_COLUMN_MAPPINGS)

def _fix(df): return df.apply(
    lambda x: pd.to_numeric(x, errors="ignore")).applymap(
    lambda x: np.NaN if x in NAN_VALUES else x)

def _parse_gct(path):
    """
    Input: path (string): full path to gct(x) file you want to parse
    Output:
    """
    # Read version and dimensions
    num_row_metadata, num_col_metadata, names = read_version_and_dims(path)

    # Read the gct file beginning with line 3
    df = pd.read_csv(
        path,
        sep='\t',
        header=list(range(num_col_metadata+1)),
        index_col=list(range(num_row_metadata+1)),
        skiprows=2,
        dtype=str,
        na_values=NAN_VALUES,
        keep_default_na=False,
    )
    df.index.set_names(names, inplace=True)
    df = _fix(df)

    idx_df = pd.DataFrame(index=df.columns)
    idx_df = _fix(idx_df.reset_index()).reset_index()
    df.columns = idx_df.set_index(df.columns.names).index

    return _normalize_index(df.transpose(), CMAP_GCT_COLUMN_MAPPINGS)

def read_version_and_dims(path):
    with open(path, 'r') as f:
        version = "CMAP_GCT%s" % f.readline().strip().lstrip('#')
        dims = list(map(int, f.readline().strip().split('\t')[2:]))

        assert version == "CMAP_GCT1.2" and len(dims) == 0 or version == "CMAP_GCT1.3" and len(dims) == 2
        if len(dims) == 0: dims = [1, 0]

        return dims + [f.readline().strip().split('\t')[:dims[0] + 1]]
