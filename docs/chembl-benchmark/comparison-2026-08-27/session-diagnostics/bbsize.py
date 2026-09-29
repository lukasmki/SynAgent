import sys; sys.path.insert(0,'src')
from FPSim2.FPSim2 import FPSim2Engine
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
import numpy as np
db='src/synagent/analogues/data/building_blocks.h5'
e=FPSim2Engine(db, in_memory_fps=True)
ids=np.asarray(e.fps[:,0]).astype(np.int64)
print('ids:', len(ids), 'range', ids.min(), ids.max(), flush=True)
smis=e.get_strings(ids)
print('smiles retrieved:', len(smis), flush=True)
hv=np.fromiter((m.GetNumHeavyAtoms() for m in (Chem.MolFromSmiles(s) for s in smis) if m is not None),
               dtype=np.int32)
print('parsed:', len(hv), flush=True)
print('heavy atoms  min %d  median %d  p99 %d  max %d' % (hv.min(), np.median(hv), np.percentile(hv,99), hv.max()))
for t in (30,40,45,50,55,62):
    print(f'  >= {t:>2} heavy atoms: {(hv>=t).sum():>7}  ({100*(hv>=t).mean():.5f}%)')
