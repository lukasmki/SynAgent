"""Could any catalog building block supply the bisindole's skeleton?
A one-step RXN1 coupling keeps most of the large reactant's atoms in the
product, so a viable partner must be close to a substructure of the target."""
import sys; sys.path.insert(0,'src')
from FPSim2.FPSim2 import FPSim2Engine
from rdkit import Chem, DataStructs, RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import rdFingerprintGenerator as fg, rdFMCS
import numpy as np

TGT='CCC1CN2CCc3c([nH]c4ccccc34)C(C(=O)OC)(c3cc4c(cc3OC)N(C)C3C(O)(C(=O)OC)C(C(=O)OC)C5(CC)C=CCN6CCC43C65)CC(C2)C1C(C)(C)C'
tm=Chem.MolFromSmiles(TGT); n_t=tm.GetNumHeavyAtoms()
gen=fg.GetMorganGenerator(radius=2, fpSize=2048); tfp=gen.GetFingerprint(tm)

e=FPSim2Engine('src/synagent/analogues/data/building_blocks.h5', in_memory_fps=True)
ids=np.asarray(e.fps[:,0]).astype(np.int64)
smis=e.get_strings(ids)

# plausible large partners: big enough to carry the skeleton, smaller than the product
cands=[]
for s in smis:
    m=Chem.MolFromSmiles(s)
    if m is None: continue
    n=m.GetNumHeavyAtoms()
    if 30 <= n < n_t:
        cands.append((s,m,n))
print(f'target {n_t} heavy atoms; plausible large partners in catalog (30..{n_t-1}): {len(cands)}', flush=True)

best_sim=(0,None); best_sub=(0,None); best_mcs=(0,None)
n_substruct=0
for s,m,n in cands:
    sim=DataStructs.TanimotoSimilarity(tfp, gen.GetFingerprint(m))
    if sim>best_sim[0]: best_sim=(sim,s)
    if tm.HasSubstructMatch(m):
        n_substruct+=1
        if n>best_sub[0]: best_sub=(n,s)
print(f'  exact substructures of the target : {n_substruct}', flush=True)
print(f'  best Tanimoto to target           : {best_sim[0]:.4f}', flush=True)

# MCS against the most similar 60, to bound the largest shared skeleton
top=sorted(cands, key=lambda c: -DataStructs.TanimotoSimilarity(tfp, gen.GetFingerprint(c[1])))[:60]
for s,m,n in top:
    try:
        r=rdFMCS.FindMCS([tm,m], timeout=3, completeRingsOnly=True)
    except Exception:
        continue
    if r and r.numAtoms>best_mcs[0]: best_mcs=(r.numAtoms,s)
print(f'  largest common substructure found : {best_mcs[0]} atoms  '
      f'({100*best_mcs[0]/n_t:.1f}% of the target)', flush=True)
print()
print(f'  a one-step route needs a partner carrying roughly {n_t-8}+ of the {n_t} atoms')
print(f'  best available shared skeleton is  {best_mcs[0]}')
