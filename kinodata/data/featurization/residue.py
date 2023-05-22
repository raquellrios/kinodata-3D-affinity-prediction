import pandas as pd

sitealign_feature_lookup = pd.DataFrame.from_dict(
    {
        "ALA": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "ARG": [3.0, 3.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        "ASN": [2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "ASP": [2.0, 0.0, 2.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        "CYS": [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "GLN": [2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "GLU": [2.0, 0.0, 2.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        "GLY": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "HIS": [2.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "ILE": [2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "LEU": [2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "LYS": [2.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        "MET": [2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "PHE": [3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "PRO": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "SER": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "THR": [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "TRP": [3.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "TYR": [3.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "VAL": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "CAF": [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "CME": [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "CSS": [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "OCY": [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "KCX": [2.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        "MSE": [2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "PHD": [2.0, 0.0, 2.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        "PTR": [3.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    },
    columns=[
        "size",
        "hbd",
        "hba",
        "charge",
        "aromatic",
        "aliphatic",
        "charge_pos",
        "charge_neg",
    ],
    orient="index",
)

amino_acid_to_int = {aa: idx for idx, aa in enumerate(sitealign_feature_lookup.index)}
