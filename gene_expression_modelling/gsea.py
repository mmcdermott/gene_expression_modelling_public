import scipy.stats
from math import factorial

from .utils import *
from .constants import *

def summarize_gene_list(genes, all_genes=LANDMARK_GENES, title="Genes", delim=', ', should_print=True):
    genes = list(genes)
    genes.sort()
    p = len(genes) / len(all_genes)
    if should_print: print("{title}:\n"
          "{sel} of {tot} total genes ({p:.2%})\n\n"
          "{genes}".format(title=title, sel=len(genes), tot=len(all_genes), p=p, genes=delim.join(genes))
         )
    return p

def enrichment_p_value(genesA, genesB, all_genes=LANDMARK_GENES):
    obs = len(set(genesA).intersection(genesB))
    A = len(genesA)
    B = len(genesB)
    tot = len(all_genes)

    exp = round(A * B / tot)

    p_le = scipy.stats.hypergeom.cdf(obs, tot, A, B)
    p_ge = scipy.stats.hypergeom.sf(obs-1, tot, A, B)
    return (obs, exp, p_le) if obs < exp else (obs, exp, p_ge)

def gene_list_enrichment_analysis(
    genesA,
    genesB,
    all_genes=LANDMARK_GENES,
    titleA="",
    titleA_short="",
    titleB="",
    titleB_short="",
    delim=', ',
    printA=True,
    printB=True,
    print_result=True,
    sig_level=0.05,
    bonferroni_correction=1,
):
    pA = summarize_gene_list(genesA, all_genes, titleA, delim, printA)
    if printA: print('\n\n')
    pB = summarize_gene_list(genesB, all_genes, titleB, delim, printB)
    if printB: print('\n\n')

    obs, exp, p_enc_raw = enrichment_p_value(genesA, genesB, all_genes)
    p_enc = p_enc_raw * bonferroni_correction
    result = ""
    if p_enc >= sig_level: result = "not significantly enriched either way"
    else:
        result = "statistically significantly mutually " + ('overenriched' if obs > exp else 'underenriched')

    both = list(set(genesA).intersection(genesB))
    both.sort()

    if print_result: print(
        "Given {tot} Genes total, with {A} {A_title} genes and {B} {B_title} genes, "
        "one should expect {exp} overlapping genes.\n"
        "We observe {obs}, which yields {test} p-value {p:.3}.\n"
        "At significance level {sig_level}, these sets are {res}.\n\n"
        "{genes}".format(
            tot=len(all_genes),
            A=len(genesA),
            B=len(genesB),
            A_title=titleA_short,
            B_title=titleB_short,
            exp=exp,
            obs=obs,
            sig_level=sig_level,
            p=p_enc,
            res=result,
            genes=delim.join(both),
            test='%shypergeometric' % ('' if bonferroni_correction == 1 else '(Bonferroni Adj.) '),
        )
    )
    return obs, exp, p_enc_raw
