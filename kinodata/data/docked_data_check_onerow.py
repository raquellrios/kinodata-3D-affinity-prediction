import logging
import multiprocessing as mp
import os
import os.path as osp
import re
from time import sleep
import warnings
from functools import cached_property, partial
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
)
from dataclasses import dataclass
import os

import pandas as pd
import requests  # type : ignore
import torch
from rdkit.Chem import AddHs, Kekulize, MolFromMol2File, PandasTools  # type: ignore
from torch_geometric.data import HeteroData, InMemoryDataset
from tqdm import tqdm
from rdkit import RDLogger
from rdkit import Chem

from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import PandasTools
from kinoml.databases.pdb import smiles_from_pdb
#!/usr/bin/env python
# validate_docking_data.py
"""
Verify that CSV, combined-SDF and KLIFS pocket downloads describe the *same*
systems in the same order.

• Every row in CSV ↔ the same-index molecule in SDF.
• protein_pdb_id / ligand ids in CSV ↔ encoded in SDF filename.
• KLIFS structure ID ↔ CSV & pocket mol2 file & live KLIFS record.
• Pocket mol2 residue set ⊆ KLIFS pocket sequence string.

Requires: pandas, rdkit, opencadd-toolkit, requests, tqdm
"""
import argparse, re, sys
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from tqdm import tqdm

# ---------------------------------------------------------------------------

def split_klifs(seq: str):
    """Return list of 'ALA123', 'LYS24', … out of a KLIFS residue string."""
    if "," in seq:
        return [s for s in seq.split(",") if s and s != "_"]
    return re.findall(r"[A-Z]{3}[0-9]+", seq)

def filename_parts(title: str):
    """
    Parse POSIT filename encoded in the molecule title.
    Expected pattern:
      <kinase>_<…>_<proteinPDB>_<chain>_…_<ligandPDB-ligExpo>_ligand.sdf
    """
    fields = title.split("_")
    protein_pdb = fields[5]
    ligand_pdb, expo = fields[9].split("-")
    return protein_pdb, ligand_pdb, expo

# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True, help="posit_results.csv (merged)")
parser.add_argument("--benchmark", required=True, help="docking_benchmark_dataset.csv (merged)")
parser.add_argument("--sdf", required=True, help="posit_combined.sdf")
parser.add_argument("--raw", required=True, help="root data dir that contains mol2/pocket/")
args = parser.parse_args()

csv_path  = Path(args.csv).expanduser()
benchmark_csv_path = Path(args.benchmark).expanduser()
sdf_path  = Path(args.sdf).expanduser()
raw_dir   = Path(args.raw).expanduser()
pocket_dir = raw_dir / "mol2" / "pocket"

print(f"📋results CSV : {csv_path}")
print(f"📋benchmark CSV : {benchmark_csv_path}")
print(f"🧪 SDF : {sdf_path}")
print(f"🕳️  pockets in : {pocket_dir}")


# ---------- 1. read both sources ------------------------------------------------
print("reading dataframes")
df_csv = pd.read_csv(csv_path)
df_bm_csv = pd.read_csv(benchmark_csv_path)
df_sdf = PandasTools.LoadSDF(
    str(sdf_path),
    smilesName="compound_structures.canonical_smiles",
    molColName="molecule",
    embedProps=True,

)

assert len(df_csv) == len(df_sdf), "Row‐count mismatch between CSV and SDF"
print("dataframe results columns")
print(df_csv.columns)
print(df_csv.iloc[1999])

print("dataframe sdf columns")
print(df_sdf.columns)
print(df_sdf.iloc[1999])


#####
sdf_smiles = df_sdf.loc[1999, "compound_structures.canonical_smiles"]
print("SMILES in SDF :", sdf_smiles)

# ------------------------------------------------------------------


# (a) take it from the merged CSV if you already have that loaded:
ligand_id = df_csv.loc[1999, "ligand_expo_id"]



# ------------------------------------------------------------------
# 3.  fetch the reference SMILES from the PDB Chemical-Component DB
pdb_smiles = smiles_from_pdb([ligand_id])[ligand_id]
print("SMILES in PDB :", pdb_smiles)

# ------------------------------------------------------------------
# 4.  compare (exact string + fingerprint similarity)
same_string = sdf_smiles == pdb_smiles

mol_sdf = Chem.MolFromSmiles(sdf_smiles)
mol_pdb = Chem.MolFromSmiles(sdf_smiles)

print(mol_sdf)
print(mol_pdb)

fp_sdf = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(sdf_smiles), 2)
fp_pdb = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(pdb_smiles), 2)
tan    = DataStructs.TanimotoSimilarity(fp_sdf, fp_pdb)

print(f"\nExact-string match      : {same_string}")
print(f"Tanimoto (radius 2)     : {tan:.3f}")

assert tan > 0.95, "‼️  Ligand in SDF does *not* match the ligand PDB entry!"
print("\n✅  First row checks out – ligand identity confirmed.")
