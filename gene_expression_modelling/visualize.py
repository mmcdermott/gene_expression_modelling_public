import matplotlib as mpl
mpl.use('Agg')

import scipy, matplotlib.pyplot as plt, pandas as pd, numpy as np, ml_toolkit.pandas_constructions as pdc
plt.style.use('ggplot')

from .data import unique_doses, restrict
from .dimensionality_reduction import pca

## Visualizing a clustered set of samples:
def visualize_parametric_2d_clustering(clusterer, save_to='temp.png', data_in_2d=None):
    if type(data_in_2d) == pd.DataFrame: data_in_2d = data_in_2d.values
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = data_in_2d[:, 0].min() - 0.1, data_in_2d[:, 0].max() + 0.1
    y_min, y_max = data_in_2d[:, 1].min() - 0.1, data_in_2d[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])

    if save_to is None: plt.ion()

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                              cmap=plt.cm.Paired,
                                         aspect='auto', origin='lower')

    plt.plot(data_in_2d[:, 0], data_in_2d[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X

    # TODO(mmd): Fix.
    if hasattr(clusterer, 'cluster_centers_'): centroids = clusterer.cluster_centers_
    elif hasattr(clusterer, 'means_'): centroids = clusterer.means_
    else: raise
    plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                                color='w', zorder=10)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    if save_to is not None: plt.savefig(save_to)
    else: plt.show()

def visualize_nonparametric_clustering(data_in_2d, labels, save_to='temp.png'):
    plt.scatter(data_in_2d[:, 0], data_in_2d[:, 1], c=labels)
    if save_to is not None: plt.savefig(save_to)
    else: plt.show()

## Visualizing One Gene:
def get_stats_str(stats):
    sample_params = stats['Sample Params']
    sample_descriptor = "{perturbagen}"
    if 'dose' in sample_params.keys() and sample_params['dose'] is not None:
        sample_descriptor += " @ {dose}"
    if 'duration' in sample_params.keys() and sample_params['duration'] is not None:
        sample_descriptor += " for {duration}h"

    sample_string = sample_descriptor.format(**sample_params)

    return "Sample: {sample}\nControl: {control}\n$p-\mathrm{{value (adj.)}} = {p_value:.4f}$".format(
        sample=sample_string, control=', '.join(stats['Control Set']), p_value=stats['p']
    )

def plot_response_curve(df, gene, perturbagens=['Lithium'], controls=None, saveTo=None, interactive=False,
    stats=None, inferred=False):
    assert not (saveTo is not None and interactive)

    data   = restrict(df, genes=[gene])
    doses  = unique_doses(data)
    labels = list(doses)
    groups = [restrict(data, doses=[dose]).values for dose in doses]

    if not controls is None:
        groups += [restrict(controls, genes=[gene]).values]
        labels.append('Control')
    positions = [1.5 * i for i in range(len(groups))]

    if saveTo is None and interactive: plt.ion()

    fig, ax = plt.subplots(1)

    ax.axhline(0)
    ax.axvline(positions[-1] - 0.75)
    ax.boxplot(groups, labels=labels, positions=positions, showmeans=True)

    if not stats is None:
        props = dict(boxstyle='round', alpha=0.5)
        stats_str = get_stats_str(stats)
        ax.text(0.05, 0.95, stats_str, transform=ax.transAxes, fontsize=14, verticalalignment='top',
            bbox=props)

    plt.ylabel('Z-score (QNorm)')
    plt.xlabel('Dosage')
    plt.title('%s%s: %s Response Pattern' % ('(Inferred) ' if inferred else '',gene, ', '.join(perturbagens)))

    if saveTo is None:
        plt.show()
    else:
        saveTo.savefig(fig)
        plt.close()

def new_plot_response_curve(
    df,
    gene,
    perturbagen  = None,
    controls     = None,
    saveTo       = None,
    interactive  = False,
    get_stats    = None,
    multiplot_by = None
):
    assert not (saveTo is not None and interactive)

    data   = restrict(df, genes=[gene])
    doses  = unique_doses(data)
    labels = list(doses)
    groups = [restrict(data, doses=[dose]).values for dose in doses]

    if not controls is None:
        groups += [restrict(controls, genes=[gene]).values]
        labels.append('Control')

    if saveTo is None and interactive: plt.ion()

    fig, ax = plt.subplots(1)

    ax.axhline(0)
    ax.boxplot(groups, labels=labels, showmeans=True)

    if not stats is None:
        props = dict(boxstyle='round', alpha=0.5)
        stats_str = get_stats_str(stats)
        ax.text(0.05, 0.95, stats_str, transform=ax.transAxes, fontsize=14, verticalalignment='top',
            bbox=props)

    plt.ylabel('Z-score (QNorm)')
    plt.xlabel('Dosage')
    plt.title('%s: %s Response Pattern' % (gene, ', '.join(perturbagens)))

    if saveTo is None:
        plt.show()
    else:
        saveTo.savefig(fig)
        plt.close()

def plot_correlational_analysis(df, save_to=None, color_by='gene', print_correlations=True, max_dims=3):
    num_samples = min(df.shape[1], max_dims)
    assert num_samples in [2, 3], "Correlational plots are only possible for 2 or 3 dimensions."
    names, samples = df.columns, df.values.T
    
    cs = pdc.get_index_levels(df, [color_by])[color_by].values.codes if color_by is not None else None
    limits = [[sample_series.min(), sample_series.max()] for sample_series in samples]
    global_lim = [min([0] + [lim[0] for lim in limits])-2, max([lim[1] for lim in limits]) + 2]
    mid = (global_lim[1] + global_lim[0])/2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') if num_samples == 3 else fig.add_subplot(111)
    ax.set_xlabel(names[0], labelpad=20)
    ax.set_ylabel(names[1], labelpad=20)
    ax.set_xlim(global_lim)
    ax.set_ylim(global_lim)
    
    if num_samples == 3:
        ax.azim = 80
        ax.plot(global_lim, global_lim, global_lim, linestyle='--', alpha=0.5)
        ax.scatter(samples[0], samples[1], samples[2], c=cs)
        ax.set_zlabel(names[2], labelpad=20)
        ax.set_zlim(global_lim)
    else: 
        ax.plot(global_lim, global_lim, linestyle='--', alpha=0.5)
        ax.scatter(samples[0], samples[1], c=cs)
    
    if print_correlations:
        # Correlation Coefficients:
        corr = scipy.stats.spearmanr(df.values)
        text = "Corr: \n%s,\n p:\n%s" % (str(corr.correlation), str(corr.pvalue))
        if num_samples == 3: ax.text(global_lim[0]-5, mid, mid, text)
        else: ax.text(global_lim[0]-5, mid, text)
    
    if save_to is None: fig.show()
    else: plt.savefig(save_to)
