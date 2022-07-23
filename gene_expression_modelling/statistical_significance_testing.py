# TODO(mmd):
# https://blog.qbaseplus.com/seven-tips-for-bio-statistical-analysis-of-gene-expression-data

import os, numpy as np, pandas as pd
from scipy.stats import rankdata, ttest_ind, ttest_ind_from_stats, spearmanr, ks_2samp, mannwhitneyu

from .data import (
    restrict, unique_doses, unique_durations, unique_genes, get_num_samples, get_num_genes,
)

from .constants import *

def get_u(ranks): return sum(ranks) - (len(ranks)*(len(ranks) + 1))/2.0
def ttest_for_one_samp(A, B, axis=0):
    assert B.size[axis] == 1, "Only supports len 1 samples."
    return ttest_ind_from_stats(
        np.mean(A, axis),
        np.std(A, axis),
        A.size[axis],
        np.mean(B, axis),
        0,
        B.size[axis],
        equal_var=True,
    )


def __mannwhitneyu_lt_20_stat(controls, samples):
    ranks = rankdata(np.concatenate((controls, samples)))
    sample_u = get_u(ranks[len(controls):])
    return min(sample_u, len(samples)*len(controls) - sample_u)

def mannwhitneyu_one_samp(controls, samples):
    sample, num_greater, n = samples[0], 0, len(controls)
    num_greater = 0
    for c in controls:
        if sample > c: num_greater += 1
    u = min(num_greater, n)
    if u != n+1 // 2:
        return u, (2.0/n+1) * (u + 1)
    return u, 1


def sorted_significant_genes(df_control, df_sample, num_tests=-1, filter_degenerate=True,
        use_nonparametric=False):
    num_samples = get_num_samples(df_sample)
    num_controls = get_num_samples(df_control)
    assert num_samples > 0 and num_controls > 0, "Not enough data!"
    num_genes = get_num_genes(df_sample)
    if num_tests == -1: num_tests = num_genes

    df = pd.DataFrame(index=df_control.columns, data={
        'control-count'  : get_num_samples(df_control),
        'sample-count'   : get_num_samples(df_sample),
        'control-mean'   : df_control.mean(axis=0),
        'control-var'    : df_control.var(axis=0),
        'sample-mean'    : df_sample.mean(axis=0),
        'sample-var'     : df_sample.var(axis=0),
        'control-median' : df_control.median(axis=0),
        'control-mad'    : df_control.mad(axis=0),
        'sample-median'  : df_sample.median(axis=0),
        'sample-mad'     : df_sample.mad(axis=0),
    })


    if use_nonparametric:
        if num_samples == 1 and num_controls > 20:
            mwutests = [mannwhitneyu_one_samp(df_control[g], df_sample[g]) for g in df_control.columns]
            df['mwutest-stat'], df['mwutest-p'] = [s for s, _ in mwutests], [p for _, p in mwutests]
        if num_samples > 20 or num_controls > 20:
            mwutests = [mannwhitneyu(df_sample[gene], df_control[gene]) for gene in df_control.columns]
            df['mwutest-stat'] = [m.statistic for m in mwutests]
            df['mwutest-p'] = [m.pvalue for m in mwutests]
        if num_samples > 5:
            kstests = [ks_2samp(df_sample[gene], df_control[gene]) for gene in df_control.columns]
            df['kstest-stat'], df['kstest-p'] = [k.statistic for k in kstests], [k.pvalue for k in kstests]
    if num_samples > 1:
        ttest            = ttest_ind(df_control, df_sample, axis=0, equal_var=False)
        df['ttest-stat'] = ttest[0]
        df['ttest-p']    = ttest[1]
    else:
        ttest            = ttest_for_one_samp(df_control, df_sample, axis=0)
        df['ttest-stat'] = ttest[0].values.reshape([-1])
        df['ttest-p']    = ttest[1].reshape([-1])

    if filter_degenerate:
        df = df[~( (df['sample-count'] > 1) & (df['sample-var'] == 0) )]

    ps = df.select(lambda colName: '-p' in colName, axis=1)
    df['p-value'] = ps.max(axis=1)
    df['ranked'] = rankdata(df['p-value'])
    df['p-value Bonferri adjusted'] = df['p-value'] * num_tests
    df['p-value FDR adjusted'] = df['p-value Bonferri adjusted']/df['ranked']

    return df.sort_values(by='p-value', ascending=True)

def significant_at(df_stats, alpha=0.05, filter_degenerate=False, only_direction=None, fold_cutoff=None,
        lfold_cutoff=None):
    max_rank = df_stats['ranked'][df_stats['p-value FDR adjusted'] < alpha].max()
    sigs = df_stats[df_stats['ranked'] <= max_rank]
    if only_direction is not None:
        if only_direction == 'DOWN': sigs = sigs[sigs['sample-median'] < sigs['control-median']]
        if only_direction == 'UP':   sigs = sigs[sigs['sample-median'] > sigs['control-median']]
    if fold_cutoff is not None:
        sigs = sigs[abs(sigs['sample-median']/sigs['control-median']) >= fold_cutoff]
    if lfold_cutoff is not None:
        sigs = sigs[abs((sigs['sample-median'] - sigs['control-median'])/sigs['control-mad']) >= lfold_cutoff]

    return sigs


def genes_at(stats_df, sig_level=0.05, only_direction=None, fold_cutoff=None, lfold_cutoff=None):
    return significant_at(
        stats_df, sig_level, only_direction=only_direction, fold_cutoff=fold_cutoff, lfold_cutoff=lfold_cutoff
    ).index.get_level_values(GENE_NAME_COL)

def _compute_bounds(groups): return [(min(group), max(group)) for group in groups]
def _ordered(l_bounds, r_bounds): return l_bounds[1] <= r_bounds[0] or r_bounds[1] <= l_bounds[0]
def _sequentially_ordered(groups):
    bounds = _compute_bounds(groups)
    return [_ordered(bounds[i-1], bounds[i]) for i in range(1, len(bounds))]
def fraction_ordered(groups):
    ordered = _sequentially_ordered(groups)
    if len(ordered) == 0: return -1.0

    return sum(ordered)/len(ordered)

def concordant_genes_by_dose(df):
    order_fractions = []
    for gene in unique_genes(df):
        per_gene = restrict(df, genes=[gene])
        order_fractions.append(fraction_ordered(
            [restrict(per_gene, doses=[dose]).values for dose in unique_doses(per_gene)]
        ))

    return pd.Series(order_fractions, index=df.columns)
