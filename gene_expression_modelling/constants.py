import numpy as np

from gene_expression_modelling.utils import *

SUBTYPE_COL      = 'subtype'
PRIMARY_SITE_COL = 'primary_site'
SAMPLE_TYPE_COL  = 'sample_type'

PERT_ID_COL         = 'pert_id'
PERT_NAME_COL       = 'pert_name'
PERT_TYPE_COL       = 'pert_type'
PERT_DOSE_COL       = 'pert_dose'
PERT_DRTN_COL       = 'pert_time'
PERT_DOSE_UNIT_COL  = 'pert_dose_unit'
PERT_DRTN_UNIT_COL  = 'pert_time_unit'
PERT_TYPE_COL       = 'pert_type'
PLATE_NAME_COL      = 'plate'

GSE_COLUMN_MAPPINGS = {
    PERT_ID_COL         : 'pert_id',
    PERT_NAME_COL       : 'pert_iname',
    PERT_DOSE_COL       : 'pert_dose',
    PERT_DRTN_COL       : 'pert_time',
    PERT_DOSE_UNIT_COL  : 'pert_dose_unit',
    PERT_DRTN_UNIT_COL  : 'pert_time_unit',
}
CMAP_GCT_COLUMN_MAPPINGS = {
    PERT_ID_COL         : 'pert_id',
    PERT_NAME_COL       : 'pert_desc',
    PERT_DOSE_COL       : 'pert_dose',
    PERT_DRTN_COL       : 'pert_time',
    PERT_DOSE_UNIT_COL  : 'pert_dose_unit',
    PERT_DRTN_UNIT_COL  : 'pert_time_unit',
}
CMAP_GCTX_COLUMN_MAPPINGS = {
    PERT_ID_COL         : 'pert_id',
    PERT_NAME_COL       : ['pert_iname', 'pert_desc'],
    PERT_DOSE_COL       : 'pert_dose',
    PERT_DRTN_COL       : 'pert_time',
    PERT_DOSE_UNIT_COL  : 'pert_dose_unit',
    PERT_DRTN_UNIT_COL  : 'pert_time_unit',
}

GENE_ID_COL   = 'pr_gene_id'
GENE_NAME_COL = 'pr_gene_symbol'

CELL_ID_COL   = 'cell_id'
CELL_TYPE_COL = 'cell_type'
CELL_IDs_DIFF_NEURONS = ['NEU', "MNEU", "NEU.KCL"]

DATA_TYPE           = np.float32
NAN_VALUES          = [
    '#N/A', 'N/A', 'NA', '#NA', 'NULL', 'NaN', '-NaN', 'nan', '-nan', '#N/A!', 'na', 'NA', 'None', '-666',
    -666, -666.0,
]

DMSO = 'DMSO'
KCL = 'KCl'
NACL = 'NaCl'
H2O = 'H2O'
PBS = 'PBS'
BUFFERS = [DMSO, KCL, NACL, H2O, PBS]



LANDMARK_GENES = depickle('landmark_genes.pkl')
LANDMARK_GENES.sort()

MITOCARTA_GENES = list(set(depickle('mitocarta_genes.pkl')).intersection(LANDMARK_GENES))
MITOCARTA_GENES.sort()
